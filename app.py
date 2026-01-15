from flask import Flask, render_template, request, jsonify
from rag_pipeline import UltimateRAG
import os

app = Flask(__name__)

# ==========================
# Configuration
# ==========================
EXCEL_PATH = "NLP_QA_Pairs12.xlsx"
PDF_DIR = "pdfs"
INDEX_PATH = "faiss_index"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("❌ GROQ_API_KEY not set. Use terminal or Render env vars.")

# ==========================
# Initialize RAG
# ==========================
print("🚀 Initializing RAG pipeline...")

rag = UltimateRAG(
    pdf_directory=PDF_DIR,
    index_path=INDEX_PATH,
    excel_path=EXCEL_PATH,
    api_key=GROQ_API_KEY
)

rag.build()
print("✅ RAG pipeline ready")

# ==========================
# Routes
# ==========================
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
        answer = rag.ask(question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"answer": f"⚠️ Error: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
