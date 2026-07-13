import os
import sys
import io
import logging

# Setup UTF-8 encoding on Windows consoles to prevent charmap errors with unicode checkmarks
if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retriever.search import FashionRetriever
from utils.config import Config

def run_evaluation():
    """Runs the evaluation queries requested by Glance."""
    # Check if index exists
    if not os.path.exists(Config.FAISS_INDEX_PATH):
        logger.error(f"FAISS index not found at '{Config.FAISS_INDEX_PATH}'. Please run indexing first.")
        return

    logger.info("Initializing FashionRetriever...")
    retriever = FashionRetriever()
    
    eval_queries = [
        "A person in a bright yellow raincoat.",
        "Professional business attire inside a modern office.",
        "Someone wearing a blue shirt sitting on a park bench.",
        "Casual weekend outfit for a city walk.",
        "A red tie and a white shirt in a formal setting."
    ]
    
    print("\n" + "="*80)
    print("RUNNING GLANCE EVALUATION QUERIES")
    print(f"Total Indexed Vectors: {retriever.vector_db.get_total_vectors()}")
    print("="*80)
    
    for query in eval_queries:
        print(f"\nQUERY: '{query}'")
        print("-" * 50)
        try:
            results = retriever.retrieve(query, top_k=3)
            if not results:
                print("No results returned.")
            for idx, res in enumerate(results):
                print(f"  {idx+1}. Score: {res['score']:.4f} | Image: {res['image_path']}")
                print(f"     Caption: {res['caption']}")
                print(f"     Attributes: Clothing: {res['clothing']} | Color: {res['color']} | Scene: {res['scene']}")
        except Exception as e:
            logger.error(f"Failed to query '{query}': {e}")
        print("-" * 50)
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    run_evaluation()
