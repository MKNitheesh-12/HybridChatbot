import os
import pandas as pd
from groq import Groq
import traceback
import shutil
import gc

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class ExcelQAProcessor:
    def __init__(self, excel_path):
        self.excel_path = excel_path
        self.qa_pairs = {}

    def load_qa_pairs(self):
        """Load Q&A pairs from Excel file"""
        try:
            df = pd.read_excel(self.excel_path)

            if "Question" not in df.columns or "Answer" not in df.columns:
                raise ValueError("Excel must have 'Question' and 'Answer' columns")

            for _, row in df.iterrows():
                q = str(row["Question"]).lower().strip()
                a = str(row["Answer"]).strip()
                self.qa_pairs[q] = a

            # Clear DataFrame from memory
            del df
            gc.collect()

            print(f"✅ Loaded {len(self.qa_pairs)} Excel Q&A pairs")
        except Exception as e:
            print(f"❌ Error loading Excel: {str(e)}")
            raise

    def get_answer(self, question):
        """Check if question exists in Excel"""
        try:
            q = question.lower().strip()
            
            # Exact match
            if q in self.qa_pairs:
                print(f"✅ Found exact match in Excel")
                return True, self.qa_pairs[q]

            # Partial match
            for k in self.qa_pairs:
                if k in q or q in k:
                    print(f"✅ Found partial match in Excel")
                    return True, self.qa_pairs[k]

            return False, ""
        except Exception as e:
            print(f"❌ Error in Excel search: {str(e)}")
            return False, ""


class UltimateRAG:
    def __init__(self, pdf_directory, index_path, excel_path, api_key):
        self.pdf_directory = pdf_directory
        self.index_path = index_path
        self.client = Groq(api_key=api_key)
        self.model_name = "llama-3.1-8b-instant"
        
        self.excel_processor = ExcelQAProcessor(excel_path)
        self.vectorstore = None
        self.retriever = None
        self.embeddings = None

    def build(self):
        """Build RAG pipeline with memory optimization"""
        try:
            print("📊 Loading Excel Q&A pairs...")
            self.excel_processor.load_qa_pairs()
            
            # Create embeddings
            self.create_embeddings()
            
            # Check if index exists
            if os.path.exists(self.index_path):
                print("📂 Found existing FAISS index...")
                try:
                    self.load_index()
                    print("✅ Existing index loaded")
                except Exception as e:
                    print(f"⚠️ Index incompatible: {str(e)}")
                    print("🔄 Rebuilding FAISS index...")
                    shutil.rmtree(self.index_path)
                    self.load_documents()
                    self.split_documents()
                    self.build_and_save_index()
                    self.load_index()
            else:
                print("📂 Building FAISS index from PDFs...")
                self.load_documents()
                self.split_documents()
                self.build_and_save_index()
                self.load_index()
                
            print("✅ RAG build complete")
        except Exception as e:
            print(f"❌ Error building RAG: {str(e)}")
            raise

    def load_documents(self):
        """Load PDF documents with memory cleanup"""
        try:
            self.documents = []

            if not os.path.exists(self.pdf_directory):
                raise ValueError(f"PDF directory '{self.pdf_directory}' not found")

            pdf_files = [f for f in os.listdir(self.pdf_directory) if f.endswith(".pdf")]
            
            if not pdf_files:
                raise ValueError(f"No PDF files found in '{self.pdf_directory}'")

            print(f"📄 Found {len(pdf_files)} PDF files")

            # Load PDFs one by one and clear memory
            for file in pdf_files:
                print(f"  Loading: {file}")
                loader = PyPDFLoader(os.path.join(self.pdf_directory, file))
                docs = loader.load()
                self.documents.extend(docs)
                
                # Clear loader from memory
                del loader, docs
                gc.collect()

            print(f"✅ Loaded {len(self.documents)} PDF pages")
        except Exception as e:
            print(f"❌ Error loading documents: {str(e)}")
            raise

    def split_documents(self):
        """Split documents into smaller chunks"""
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,  # Reduced from 1000
                chunk_overlap=150  # Reduced from 200
            )
            self.chunks = splitter.split_documents(self.documents)
            print(f"✂️ Created {len(self.chunks)} chunks")
            
            # Clear documents after chunking
            del self.documents
            gc.collect()
        except Exception as e:
            print(f"❌ Error splitting documents: {str(e)}")
            raise

    def create_embeddings(self):
        """Create lightweight embeddings model"""
        try:
            print("🔮 Creating embeddings model...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            test_embed = self.embeddings.embed_query("test")
            print(f"✅ Embeddings ready (dim: {len(test_embed)})")
        except Exception as e:
            print(f"❌ Error creating embeddings: {str(e)}")
            raise

    def build_and_save_index(self):
        """Build and save FAISS index"""
        try:
            print("🏗️ Building FAISS index...")
            self.vectorstore = FAISS.from_documents(self.chunks, self.embeddings)
            self.vectorstore.save_local(self.index_path)
            
            # Clear chunks after indexing
            del self.chunks
            gc.collect()
            
            print(f"💾 FAISS index saved to {self.index_path}")
        except Exception as e:
            print(f"❌ Error building FAISS index: {str(e)}")
            raise

    def load_index(self):
        """Load FAISS index"""
        try:
            print(f"📥 Loading FAISS index from {self.index_path}...")
            self.vectorstore = FAISS.load_local(
                self.index_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": 3}  # Reduced from 4
            )
            
            # Test retrieval
            test_docs = self.retriever.invoke("test query")
            
            print("✅ FAISS index loaded successfully")
        except Exception as e:
            print(f"❌ Error loading FAISS index: {str(e)}")
            raise

    def ask(self, question):
        """Answer a question with memory management"""
        try:
            print(f"🤔 Processing: {question}")
            
            # Check Excel first
            found, answer = self.excel_processor.get_answer(question)
            if found:
                return answer

            # RAG pipeline
            print("🔍 Searching in vector store...")
            docs = self.retriever.invoke(question)
            print(f"📚 Found {len(docs)} relevant documents")
            
            if not docs:
                return "I couldn't find relevant information. Please try rephrasing."
            
            context = "\n\n".join(d.page_content for d in docs)

            prompt = f"""You are a helpful AI assistant. Answer based on the context below.

Context:
{context}

Question: {question}

Answer (be concise):"""

            print("🤖 Calling Groq API...")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=512
            )

            answer = response.choices[0].message.content.strip()
            
            # Clean up
            del docs, context, response
            gc.collect()
            
            return answer
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            raise