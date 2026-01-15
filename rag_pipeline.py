import os
import pandas as pd
from groq import Groq

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# ============================================================
# 📊 EXCEL Q&A PROCESSOR
# ============================================================

class ExcelQAProcessor:
    def __init__(self, excel_path):
        self.excel_path = excel_path
        self.qa_pairs = {}

    def load_qa_pairs(self):
        df = pd.read_excel(self.excel_path)

        if "Question" not in df.columns or "Answer" not in df.columns:
            raise ValueError("Excel must have Question & Answer columns")

        for _, row in df.iterrows():
            q = str(row["Question"]).lower().strip()
            a = str(row["Answer"]).strip()
            self.qa_pairs[q] = a

        print(f"✅ Loaded {len(self.qa_pairs)} Excel Q&A pairs")

    def get_answer(self, question):
        q = question.lower().strip()
        if q in self.qa_pairs:
            return True, self.qa_pairs[q]

        for k in self.qa_pairs:
            if k in q or q in k:
                return True, self.qa_pairs[k]

        return False, ""


# ============================================================
# 🧠 ULTIMATE RAG
# ============================================================

class UltimateRAG:
    def __init__(self, pdf_directory, index_path, excel_path, api_key):
        self.pdf_directory = pdf_directory
        self.index_path = index_path
        self.client = Groq(api_key=api_key)
        self.model_name = "llama-3.1-8b-instant"

        self.excel_processor = ExcelQAProcessor(excel_path)

    def build(self):
        self.excel_processor.load_qa_pairs()
        self.load_documents()
        self.split_documents()
        self.create_embeddings()

        if not os.path.exists(self.index_path):
            self.build_and_save_index()

        self.load_index()

    def load_documents(self):
        self.documents = []

        for file in os.listdir(self.pdf_directory):
            if file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(self.pdf_directory, file))
                self.documents.extend(loader.load())

        if not self.documents:
            raise ValueError("No documents found")

        print(f"📂 Loaded {len(self.documents)} documents")

    def split_documents(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=400)
        self.chunks = splitter.split_documents(self.documents)

    def create_embeddings(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def build_and_save_index(self):
        self.vectorstore = FAISS.from_documents(self.chunks, self.embeddings)
        self.vectorstore.save_local(self.index_path)

    def load_index(self):
        self.vectorstore = FAISS.load_local(
            self.index_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 6})

    def ask(self, question):
        found, answer = self.excel_processor.get_answer(question)
        if found:
            return answer

        docs = self.retriever.invoke(question)
        context = "\n".join(d.page_content for d in docs)

        prompt = f"""
Use only the context below.

Context:
{context}

Question:
{question}

Answer:
"""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024
        )

        return response.choices[0].message.content
