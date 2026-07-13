import os
import sys
import logging
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add parent directory to path to import configuration and retrieval modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import Config
from retriever.search import FashionRetriever

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = "glance_fashion_retrieval_secret_key"

# Global retriever instance, lazily loaded
retriever = None

def get_retriever():
    """Lazily load the FashionRetriever to prevent overhead on application start."""
    global retriever
    if retriever is None:
        logger.info("Initializing FashionRetriever in Flask application...")
        # Since GTX 1650 has 4GB VRAM, loading might take a few seconds
        retriever = FashionRetriever()
        logger.info("FashionRetriever initialized successfully.")
    return retriever

@app.route("/")
def home():
    """Renders the Home search page."""
    # Check if the FAISS index exists
    index_exists = os.path.exists(Config.FAISS_INDEX_PATH)
    total_images = 0
    if index_exists:
        try:
            r = get_retriever()
            total_images = r.vector_db.get_total_vectors()
        except Exception as e:
            logger.error(f"Error checking index size: {e}")
            
    return render_template("index.html", index_exists=index_exists, total_images=total_images)

@app.route("/search", methods=["POST"])
def search():
    """Handles search query form submission and redirects to results."""
    query = request.form.get("query", "").strip()
    if not query:
        return redirect(url_for("home"))
    
    # Redirect to results page with query parameter
    return redirect(url_for("results", query=query))

@app.route("/results")
def results():
    """Performs semantic search and displays retrieval results."""
    query = request.args.get("query", "").strip()
    if not query:
        return redirect(url_for("home"))
        
    try:
        r = get_retriever()
        # Retrieve top 12 results for a beautiful grid (Bootstrap col-md-4 or col-md-3)
        search_results = r.retrieve(query, top_k=12)
    except Exception as e:
        logger.error(f"Search retrieval error for query '{query}': {e}", exc_info=True)
        search_results = []
        
    return render_template(
        "index.html", 
        query=query, 
        results=search_results, 
        index_exists=True,
        total_images=r.vector_db.get_total_vectors() if retriever else 0
    )

@app.route("/api/search", methods=["GET"])
def api_search():
    """API endpoint for AJAX search queries, returning JSON."""
    query = request.args.get("query", "").strip()
    top_k = int(request.args.get("top_k", 10))
    if not query:
        return jsonify({"error": "Empty query"}), 400
        
    try:
        r = get_retriever()
        search_results = r.retrieve(query, top_k=top_k)
        return jsonify({"query": query, "results": search_results})
    except Exception as e:
        logger.error(f"API search error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/dataset/images/<path:filename>")
def serve_dataset_image(filename):
    """Serves raw images directly from the local dataset folder."""
    return send_from_directory(Config.DATASET_DIR, filename)

if __name__ == "__main__":
    # Ensure configuration directories exist
    Config.create_dirs()
    
    # Start local development server
    logger.info("Starting Flask application...")
    app.run(host="127.0.0.1", port=5000, debug=False)
