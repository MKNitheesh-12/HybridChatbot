from flask import Flask, render_template, request, jsonify
from rag_pipeline import UltimateRAG
import os
from dotenv import load_dotenv
import traceback
import gc

load_dotenv()

app = Flask(__name__)

# Configuration
EXCEL_PATH = "NLP_QA_Pairs12.xlsx"
PDF_DIR = "pdfs"
INDEX_PATH = "faiss_index"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("❌ GROQ_API_KEY not set in environment variables")

# Global RAG instance (lazy loaded)
rag = None

def get_rag():
    """Lazy load RAG only when needed to save memory"""
    global rag
    if rag is None:
        print("🚀 Initializing RAG pipeline...")
        try:
            rag = UltimateRAG(
                pdf_directory=PDF_DIR,
                index_path=INDEX_PATH,
                excel_path=EXCEL_PATH,
                api_key=GROQ_API_KEY
            )
            rag.build()
            gc.collect()
            print("✅ RAG pipeline ready")
        except Exception as e:
            print(f"❌ Error initializing RAG: {e}")
            raise
    return rag

@app.route("/")
def home():
    """Render the main chatbot interface"""
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """Handle chat requests from the user"""
    try:
        data = request.get_json()
        question = data.get("question", "").strip()

        if not question:
            return jsonify({"answer": "⚠️ Please enter a valid question."}), 400

        print(f"📥 Received question: {question}")
        
        # Get RAG instance and process question
        rag_instance = get_rag()
        answer = rag_instance.ask(question)
        
        # Clean up memory
        gc.collect()
        
        print(f"✅ Response generated successfully")
        return jsonify({"answer": answer})
        
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"❌ Error processing request:\n{error_details}")
        return jsonify({
            "answer": f"⚠️ An error occurred while processing your question. Please try again."
        }), 500

@app.route("/health")
def health():
    """Health check endpoint for monitoring"""
    try:
        # Check if essential files exist
        checks = {
            "excel": os.path.exists(EXCEL_PATH),
            "pdfs": os.path.exists(PDF_DIR),
            "index": os.path.exists(INDEX_PATH),
            "api_key": bool(GROQ_API_KEY)
        }
        
        all_healthy = all(checks.values())
        
        return jsonify({
            "status": "healthy" if all_healthy else "degraded",
            "checks": checks
        }), 200 if all_healthy else 503
        
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 503

@app.route("/stats")
def stats():
    """Get system statistics"""
    try:
        rag_instance = get_rag()
        
        stats_data = {
            "excel_qa_pairs": len(rag_instance.excel_qa_pairs) if rag_instance.excel_qa_pairs else 0,
            "documents_loaded": len(rag_instance.documents) if hasattr(rag_instance, 'documents') else 0,
            "chunks_created": len(rag_instance.chunks) if hasattr(rag_instance, 'chunks') else 0,
            "model": rag_instance.model_name,
            "status": "ready"
        }
        
        return jsonify(stats_data)
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # Use debug=False in production
    app.run(host="0.0.0.0", port=port, debug=False)