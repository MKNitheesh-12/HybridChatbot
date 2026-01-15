from flask import Flask, render_template, request, jsonify
from rag_pipeline import UltimateRAG
import os
from dotenv import load_dotenv
import traceback  # NEW - for detailed error tracking

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# ==========================
# Configuration
# ==========================
EXCEL_PATH = "NLP_QA_Pairs12.xlsx"
PDF_DIR = "pdfs"
INDEX_PATH = "faiss_index"

# Get API key from environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("❌ GROQ_API_KEY not set. Create a .env file with your API key.")

# ==========================
# Initialize RAG (Lazy Loading)
# ==========================
rag = None

def get_rag():
    """Lazy load RAG to reduce memory usage"""
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
        print("✅ RAG pipeline ready")
    return rag

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
        print(f"📥 Question received: {question}")  # NEW - log question
        rag_instance = get_rag()
        print("🔍 Processing with RAG...")  # NEW
        answer = rag_instance.ask(question)
        print(f"✅ Answer generated: {answer[:100]}...")  # NEW - log first 100 chars
        return jsonify({"answer": answer})
    except Exception as e:
        # Print full error traceback
        error_details = traceback.format_exc()
        print(f"❌ Error occurred:\n{error_details}")  # NEW - detailed error
        return jsonify({"answer": f"⚠️ Error: {str(e)}"}), 500

@app.route("/health")
def health():
    """Health check endpoint for Render"""
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)  # Added debug=True