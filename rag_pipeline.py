import os
import pandas as pd
from groq import Groq
from typing import Tuple
import gc
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


class ExcelQAProcessor:
    """Process and search Q&A pairs from Excel sheet using NLP similarity"""
    
    def __init__(self, excel_path):
        self.excel_path = excel_path
        self.qa_pairs = {}
        self.questions_list = []
        
    def load_qa_pairs(self):
        """Load Q&A pairs from Excel file"""
        try:
            if not os.path.exists(self.excel_path):
                print(f"⚠️ Excel file not found at: {self.excel_path}")
                return {}

            df = pd.read_excel(self.excel_path)

            # Check required columns
            required_columns = ['Question', 'Answer']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")

            # Create dictionary of Q&A pairs
            for _, row in df.iterrows():
                question = str(row['Question']).strip()
                answer = str(row['Answer']).strip()
                if question and answer and question != 'nan' and answer != 'nan':
                    self.qa_pairs[question.lower()] = answer
                    self.questions_list.append(question.lower())

            print(f"✅ Loaded {len(self.qa_pairs)} Q&A pairs from Excel")
            return self.qa_pairs

        except Exception as e:
            print(f"⚠️ Error loading Excel file: {e}")
            print("   Continuing with RAG-only mode...")
            return {}

    def calculate_similarity(self, query: str, target: str) -> float:
        """Calculate simple word overlap similarity (TF-IDF-like approach)"""
        query_words = set(query.lower().split())
        target_words = set(target.lower().split())
        
        if not query_words or not target_words:
            return 0.0
        
        # Jaccard similarity
        intersection = query_words.intersection(target_words)
        union = query_words.union(target_words)
        
        return len(intersection) / len(union) if union else 0.0

    def get_answer(self, question: str, threshold: float = 0.3) -> Tuple[bool, str]:
        """
        Search for answer using NLP similarity (cosine similarity approximation)
        Returns: (found: bool, answer: str)
        """
        if not self.qa_pairs:
            return False, ""

        question_lower = question.lower().strip()

        # 1. Exact match
        if question_lower in self.qa_pairs:
            return True, self.qa_pairs[question_lower]

        # 2. Find best match using similarity
        best_match = None
        best_score = 0.0
        
        for excel_question in self.questions_list:
            similarity = self.calculate_similarity(question_lower, excel_question)
            
            if similarity > best_score:
                best_score = similarity
                best_match = excel_question
        
        # 3. Return if similarity is above threshold
        if best_score >= threshold and best_match:
            print(f"📊 Excel match found (similarity: {best_score:.2f})")
            return True, self.qa_pairs[best_match]

        return False, ""


class UltimateRAG:
    """
    Complete RAG pipeline with:
    - Excel Q&A search using NLP similarity
    - Vector index retrieval
    - LLM-based completion
    """
    
    def __init__(self, pdf_directory="pdfs", index_path="faiss_index",
                 excel_path=None, api_key=None):
        self.pdf_directory = pdf_directory
        self.index_path = index_path
        self.excel_path = excel_path
        self.client = Groq(api_key=api_key)
        self.model_name = "llama-3.1-8b-instant"
        self.excel_processor = None
        self.excel_qa_pairs = {}
        self.documents = []
        self.chunks = []

        if excel_path and os.path.exists(excel_path):
            self.excel_processor = ExcelQAProcessor(excel_path)

    def load_excel_qa(self):
        """Load pre-defined Q&A from Excel"""
        if not self.excel_processor:
            print("\n⚠️ No Excel file configured. Using RAG only.")
            return

        print("\n📊 Loading Excel Q&A pairs...")
        self.excel_qa_pairs = self.excel_processor.load_qa_pairs()

        if self.excel_qa_pairs:
            print("Sample questions from Excel:")
            for i, q in enumerate(list(self.excel_qa_pairs.keys())[:3], 1):
                print(f"  {i}. {q[:60]}...")

    def load_documents(self):
        """Load documents from PDF directory"""
        self.documents = []
        print("\n📂 Loading documents...")

        if not os.path.exists(self.pdf_directory):
            raise ValueError(f"❌ PDF directory not found: {self.pdf_directory}")

        pdf_files = [f for f in os.listdir(self.pdf_directory) if f.endswith(".pdf")]
        
        if not pdf_files:
            raise ValueError(f"❌ No PDF files found in {self.pdf_directory}")

        for pdf in pdf_files:
            try:
                loader = PyPDFLoader(os.path.join(self.pdf_directory, pdf))
                docs = loader.load()
                docs = [d for d in docs if d.page_content.strip()]

                if docs:
                    self.documents.extend(docs)
                    print(f"✓ Loaded {pdf} ({len(docs)} pages)")
                else:
                    print(f"⚠️ {pdf} is empty, skipping...")

            except Exception as e:
                print(f"⚠️ Error loading {pdf}: {e}")

        if not self.documents:
            raise ValueError("❌ No valid documents found. Cannot build RAG.")

        print(f"\n✅ Total documents loaded: {len(self.documents)}")

    def split_documents(self):
        """Split documents into chunks for vector indexing"""
        print("\n✂️ Splitting documents into chunks...")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1600,
            chunk_overlap=400,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        self.chunks = splitter.split_documents(self.documents)
        self.chunks = [c for c in self.chunks if c.page_content.strip()]

        if not self.chunks:
            raise ValueError("❌ No chunks created. Check document content.")

        print(f"✅ Total chunks created: {len(self.chunks)}")
        print(f"   Average chunk size: {sum(len(c.page_content) for c in self.chunks) // len(self.chunks)} chars")

    def create_embeddings(self):
        """Create embeddings model for vector indexing"""
        print("\n🧮 Creating embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            encode_kwargs={"normalize_embeddings": True}
        )
        print("✅ Embeddings model ready")

    def build_and_save_index(self):
        """Build FAISS vector index and save to disk"""
        print("\n💾 Building FAISS index...")

        self.vectorstore = FAISS.from_documents(
            documents=self.chunks,
            embedding=self.embeddings
        )

        self.vectorstore.save_local(self.index_path)
        print(f"✅ FAISS index saved to: {self.index_path}")

    def load_index(self):
        """Load existing FAISS index from disk"""
        print("\n📦 Loading FAISS index from disk...")

        if not os.path.exists(self.index_path):
            print(f"⚠️ Index not found at {self.index_path}. Building new index...")
            if not hasattr(self, 'chunks') or not self.chunks:
                self.load_documents()
                self.split_documents()
            self.build_and_save_index()

        try:
            self.vectorstore = FAISS.load_local(
                self.index_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )

            self.retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 8, "fetch_k": 16}
            )

            print("✅ FAISS index loaded and retriever ready")
            
        except Exception as e:
            print(f"⚠️ Error loading index: {e}")
            print("🔄 Rebuilding FAISS index...")
            
            import shutil
            if os.path.exists(self.index_path):
                shutil.rmtree(self.index_path)
            
            if not hasattr(self, 'chunks') or not self.chunks:
                self.load_documents()
                self.split_documents()
            
            self.build_and_save_index()
            
            self.vectorstore = FAISS.load_local(
                self.index_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            self.retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 8, "fetch_k": 16}
            )
            print("✅ Index rebuilt and loaded successfully")

    def clean_response(self, text: str) -> str:
        """Remove code artifacts from LLM response"""
        # Remove code blocks
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`[^`]+`', '', text)
        
        # Remove programming keywords
        keywords = ['python', 'print(', 'def ', 'import ', 'return ', 'result']
        for keyword in keywords:
            text = text.replace(keyword, '')
        
        # Clean whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = ' '.join(text.split())
        
        return text.strip()

    def ask(self, question: str) -> str:
        """
        Process question with priority:
        1. Excel Q&A (using NLP similarity)
        2. Vector index retrieval + LLM completion
        """

        # Step 1: Check Excel Q&A first (using similarity matching)
        if self.excel_processor:
            found, excel_answer = self.excel_processor.get_answer(question)
            if found:
                return f"\n\n{excel_answer}"

        # Step 2: Use RAG pipeline
        if not hasattr(self, 'retriever'):
            return "❌ Error: RAG system not initialized. Please contact support."

        try:
            # Retrieve relevant documents
            docs = self.retriever.invoke(question)

            if not docs:
                return "⚠️ No relevant information found in the knowledge base for this question."

            # Combine context from top documents
            context = "\n\n".join(d.page_content for d in docs[:6])

            # Create prompt for LLM
            prompt = f"""You are a helpful AI assistant specialized in Natural Language Processing (NLP). 
Answer the question based ONLY on the context provided below.

CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
- Provide a clear, detailed answer in natural language
- Use only information from the context
- DO NOT include code snippets, function names, or technical syntax
- Explain concepts conversationally without showing code
- If the context contains code, describe what it does in plain English
- Be professional and comprehensive
- If the context doesn't contain enough information, say so honestly

ANSWER:"""

            # Get LLM completion
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=2048
            )

            answer = response.choices[0].message.content
            cleaned_answer = self.clean_response(answer)

            return f"\n\n{cleaned_answer}"

        except Exception as e:
            print(f"❌ Error in ask(): {e}")
            return f"⚠️ Error generating response. Please try again or rephrase your question."

    def build(self):
        """Build complete RAG pipeline"""
        print("\n" + "="*80)
        print("BUILDING HYBRID RAG PIPELINE")
        print("="*80)

        try:
            # Load Excel Q&A
            self.load_excel_qa()

            # Build vector index
            self.load_documents()
            self.split_documents()
            self.create_embeddings()
            
            # Try to load existing index, build if doesn't exist
            if os.path.exists(self.index_path):
                self.load_index()
            else:
                self.build_and_save_index()
                self.load_index()

            print("\n" + "="*80)
            print("✅ PIPELINE BUILD COMPLETE")
            print("="*80)
            print(f"   📊 Excel Q&A pairs: {len(self.excel_qa_pairs)}")
            print(f"   📄 Documents loaded: {len(self.documents)}")
            print(f"   ✂️  Chunks created: {len(self.chunks)}")
            print(f"   🗂️  Vector index: {self.index_path}")
            print("="*80)

            # Clear memory
            gc.collect()

        except Exception as e:
            print(f"\n❌ Error building pipeline: {e}")
            raise