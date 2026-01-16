import os
import pandas as pd
from groq import Groq
from typing import Tuple
import gc

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


class ExcelQAProcessor:
    def __init__(self, excel_path):
        self.excel_path = excel_path
        self.qa_pairs = {}

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

            print(f"✅ Loaded {len(self.qa_pairs)} Q&A pairs from Excel")
            return self.qa_pairs

        except Exception as e:
            print(f"⚠️ Error loading Excel file: {e}")
            print("   Continuing with RAG-only mode...")
            return {}

    def get_answer(self, question: str) -> Tuple[bool, str]:
        """Check if question exists in Excel Q&A and return answer"""
        if not self.qa_pairs:
            return False, ""

        question_lower = question.lower().strip()

        # Exact match
        if question_lower in self.qa_pairs:
            return True, self.qa_pairs[question_lower]

        # Partial match (check if any Excel question is contained in user question)
        for excel_question in self.qa_pairs.keys():
            if excel_question in question_lower or question_lower in excel_question:
                return True, self.qa_pairs[excel_question]

        return False, ""


class UltimateRAG:
    def __init__(self, pdf_directory="pdfs", index_path="faiss_index",
                 excel_path=None, api_key=None):
        self.pdf_directory = pdf_directory
        self.index_path = index_path
        self.excel_path = excel_path
        self.client = Groq(api_key=api_key)
        self.model_name = "llama-3.1-8b-instant"
        self.excel_processor = None
        self.excel_qa_pairs = {}

        if excel_path and os.path.exists(excel_path):
            self.excel_processor = ExcelQAProcessor(excel_path)

    # --------------------------------------------------------
    # Load Excel Q&A Pairs
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Load Documents (PDF + TXT fallback)
    # --------------------------------------------------------
    def load_documents(self):
        self.documents = []
        print("\n📂 Loading documents...")

        # Load PDFs
        if os.path.exists(self.pdf_directory):
            pdf_files = [f for f in os.listdir(self.pdf_directory) if f.endswith(".pdf")]

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

    # --------------------------------------------------------
    # Chunking - EXACTLY AS IN COLAB
    # --------------------------------------------------------
    def split_documents(self):
        print("\n✂️ Splitting documents into chunks...")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1600,      # ← COLAB VALUE
            chunk_overlap=400,    # ← COLAB VALUE
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        self.chunks = splitter.split_documents(self.documents)
        self.chunks = [c for c in self.chunks if c.page_content.strip()]

        if not self.chunks:
            raise ValueError("❌ No chunks created. Check document content.")

        print(f"✅ Total chunks created: {len(self.chunks)}")
        print(f"   Average chunk size: {sum(len(c.page_content) for c in self.chunks) // len(self.chunks)} chars")

    # --------------------------------------------------------
    # Embeddings - EXACTLY AS IN COLAB
    # --------------------------------------------------------
    def create_embeddings(self):
        print("\n🧮 Creating embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",  # ← COLAB MODEL
            encode_kwargs={"normalize_embeddings": True}
        )
        print("✅ Embeddings model ready")

    # --------------------------------------------------------
    # Vector Store (SAVE + LOAD)
    # --------------------------------------------------------
    def build_and_save_index(self):
        print("\n💾 Building FAISS index...")

        self.vectorstore = FAISS.from_documents(
            documents=self.chunks,
            embedding=self.embeddings
        )

        self.vectorstore.save_local(self.index_path)
        print(f"✅ FAISS index saved to: {self.index_path}")

    def load_index(self):
        print("\n📦 Loading FAISS index from disk...")

        if not os.path.exists(self.index_path):
            raise ValueError(f"❌ Index not found at {self.index_path}. Run build() first.")

        try:
            self.vectorstore = FAISS.load_local(
                self.index_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )

            # ← COLAB RETRIEVAL CONFIG
            self.retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 8, "fetch_k": 16}
            )

            print("✅ FAISS index loaded and retriever ready")
            
        except Exception as e:
            print(f"⚠️ Error loading index (incompatible model): {e}")
            print("🔄 Rebuilding FAISS index with new embeddings model...")
            
            # Delete incompatible index
            import shutil
            if os.path.exists(self.index_path):
                shutil.rmtree(self.index_path)
            
            # Rebuild from scratch
            if not hasattr(self, 'chunks'):
                self.load_documents()
                self.split_documents()
            
            self.build_and_save_index()
            
            # Try loading again
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

    # --------------------------------------------------------
    # Clean Response Function
    # --------------------------------------------------------
    def clean_response(self, text):
        """Remove code artifacts and technical syntax from response"""
        import re
        
        # Remove code blocks (```...```)
        text = re.sub(r'```[\s\S]*?```', '', text)
        
        # Remove inline code (`...`)
        text = re.sub(r'`[^`]+`', '', text)
        
        # Remove common programming keywords
        keywords_to_remove = [
            'python', 'print(', 'result', 'return', 'def ', 'import ',
            'from ', '.py', '()', 'console.log', 'function', 'var ', 'let ', 'const '
        ]
        
        for keyword in keywords_to_remove:
            text = text.replace(keyword, '')
        
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Clean up extra spaces
        text = ' '.join(text.split())
        
        return text.strip()

    # --------------------------------------------------------
    # Enhanced Ask Function - EXACTLY AS IN COLAB
    # --------------------------------------------------------
    def ask(self, question):
        """Ask question with priority: Excel Q&A > RAG"""

        # 1. First check Excel Q&A
        if self.excel_processor:
            found, excel_answer = self.excel_processor.get_answer(question)
            if found:
                return f"\n\n{excel_answer}"

        # 2. If not found in Excel, use RAG
        if not hasattr(self, 'retriever'):
            return "❌ Error: RAG system not initialized. Please run build() first."

        try:
            docs = self.retriever.invoke(question)

            if not docs:
                return "⚠️ No relevant information found in the knowledge base for this question."

            context = "\n\n".join(d.page_content for d in docs[:6])

            prompt = f"""You are a helpful AI assistant. Answer the question based ONLY on the context provided below.

CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
- Provide a clear, detailed answer in natural language
- Use only information from the context
- DO NOT include code snippets, function names, or technical syntax
- Remove programming keywords like "python", "print", "result", variable names
- Explain concepts conversationally without showing code examples
- If the context contains code, describe what it does in plain English
- Be concise but comprehensive
- Format your answer as a professional explanation

ANSWER:"""

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=2048
            )

            return f"\n\n{response.choices[0].message.content}"

        except Exception as e:
            return f"❌ Error generating response: {str(e)}"

    # --------------------------------------------------------
    # Build Complete Pipeline - EXACTLY AS IN COLAB
    # --------------------------------------------------------
    def build(self):
        """Build complete RAG pipeline with Excel integration"""
        print("\n" + "="*80)
        print("PHASE 2: BUILDING RAG PIPELINE WITH EXCEL INTEGRATION")
        print("="*80)

        try:
            # Load Excel Q&A
            self.load_excel_qa()

            # Build RAG components
            self.load_documents()
            self.split_documents()
            self.create_embeddings()
            self.build_and_save_index()
            self.load_index()

            print("\n" + "="*80)
            print("✅ COMPLETE RAG PIPELINE BUILT SUCCESSFULLY")
            print("="*80)
            print(f"   - Excel Q&A pairs: {len(self.excel_qa_pairs)}")
            print(f"   - Document chunks: {len(self.chunks)}")
            print(f"   - Vector index: {self.index_path}")
            print("="*80)

        except Exception as e:
            print(f"\n❌ Error building pipeline: {e}")
            raise