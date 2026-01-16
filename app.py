from flask import Flask, render_template, request, jsonify
from rag_pipeline import UltimateRAG
import os
from dotenv import load_dotenv
import traceback
import gc  # Garbage collection

load_dotenv()

app = Flask(__name__)

# Configuration
EXCEL_PATH = "NLP_QA_Pairs12.xlsx"
PDF_DIR = "pdfs"
INDEX_PATH = "faiss_index"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("❌ GROQ_API_KEY not set")

# Don't initialize RAG at startup
rag = None

def get_rag():
    """Lazy load RAG only when needed"""
    global rag
    if rag is None:
        print("🚀 Initializing RAG pipeline...")
        rag = UltimateRAG(
            pdf_directory=PDF_DIR,
            index_path=INDEX_PATH,
            excel_path=EXCEL_PATH,
            api_key=GROQ_API_KEY
        )
        rag.build()
        # Force garbage collection
        gc.collect()
        print("✅ RAG pipeline ready")
    return rag

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"answer": "Please enter a valid question."})

    try:
        print(f"📥 Question: {question}")
        rag_instance = get_rag()
        answer = rag_instance.ask(question)
        
        # Clear memory after response
        gc.collect()
        
        return jsonify({"answer": answer})
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"❌ Error:\n{error_details}")
        return jsonify({"answer": f"⚠️ Error: {str(e)}"}), 500

@app.route("/health")
def health():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # Remove debug=True in production
    app.run(host="0.0.0.0", port=port, debug=False)