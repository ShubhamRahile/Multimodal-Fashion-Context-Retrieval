import os
import sys
import logging
from typing import List, Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import Config
from indexer.feature_extractor import FeatureExtractor
from indexer.vector_store import VectorStore
from retriever.query_encoder import QueryEncoder
from retriever.ranking import RankingEngine

logger = logging.getLogger(__name__)

class FashionRetriever:
    """Orchestrates query encoding, FAISS vector search, and multi-similarity re-ranking with fallback safety."""

    def __init__(self, feature_extractor: FeatureExtractor = None):
        """Initializes retriever with exception safety for missing index files."""
        if feature_extractor is not None:
            self.extractor = feature_extractor
        else:
            self.extractor = FeatureExtractor()
            
        self.encoder = QueryEncoder(self.extractor)
        self.ranker = RankingEngine()
        
        try:
            self.vector_db = VectorStore()
            self.index_loaded = True
            logger.info("Retriever successfully initialized Vector Store index.")
        except Exception as e:
            logger.warning(f"Vector Store index not loaded (database is likely empty or indexing in progress): {e}")
            self.vector_db = None
            self.index_loaded = False

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Performs search queries, returning empty list if index is not initialized."""
        if not query or not query.strip():
            return []
            
        # Hot-reload check: if index wasn't loaded on startup, check if it's available now
        if not self.index_loaded or self.vector_db is None or self.vector_db.index is None:
            try:
                self.vector_db = VectorStore()
                if self.vector_db.index is not None:
                    self.index_loaded = True
                    logger.info("Retriever successfully hot-loaded newly generated FAISS index.")
                else:
                    return []
            except Exception:
                return []

        logger.info(f"Retrieving advanced Top-{top_k} results for: '{query}'")
        
        # 1. Encode query
        query_emb = self.encoder.encode_query(query)
        
        # 2. Retrieve candidate pool
        candidate_k = max(top_k * 3, 20)
        raw_results = self.vector_db.search(query_emb, top_k=candidate_k)
        
        # 3. Apply Re-ranking
        reranked = self.ranker.rerank_results(query, query_emb, raw_results)
        
        # 4. Format outputs with explainable match attributes
        output_results = []
        query_lower = query.lower()
        
        for score, meta in reranked[:top_k]:
            clothing = meta.get("detected_clothing", "apparel")
            color = meta.get("detected_color", "multicolor")
            style = meta.get("style", "casual")
            scene = meta.get("scene", "outdoor")
            activity = meta.get("activity", "standing")
            environment = meta.get("environment", "outdoor")
            raw_caption = meta.get("caption", "")

            matches = []
            
            if any(w in query_lower for w in clothing.split()):
                matches.append(f"✓ {clothing.title()}")
                
            if any(w in query_lower for w in color.split()):
                matches.append(f"✓ {color.title()}")
                
            if any(w in query_lower for w in style.split()):
                matches.append(f"✓ {style.title()}")
            elif "weekend" in query_lower and style in ["casual", "streetwear"]:
                matches.append(f"✓ {style.title()} style")
            elif "office" in query_lower and style in ["business casual", "formal", "minimalist"]:
                matches.append(f"✓ {style.title()} style")
            elif "vacation" in query_lower and style in ["casual", "sporty"]:
                matches.append(f"✓ Vacation {style.title()}")
                
            if any(w in query_lower for w in scene.split()):
                matches.append(f"✓ {scene.title()}")
            elif "walk" in query_lower and scene in ["street", "park"]:
                matches.append(f"✓ {scene.title()} walk")
                
            if any(w in query_lower for w in activity.split()):
                matches.append(f"✓ {activity.title()}")
                
            if any(w in query_lower for w in environment.split()):
                matches.append(f"✓ {environment.title()}")
            elif "city" in query_lower and environment == "outdoor":
                matches.append("✓ Outdoor context")

            if not matches:
                matches = [f"✓ {clothing.title()}", f"✓ {color.title()}", f"✓ {style.title()} style"]

            explain_str = " | ".join(matches)
            fused_caption = f"{raw_caption}\n\nMatched: {explain_str}"
            
            output_results.append({
                "image_path": meta["image_path"],
                "score": score,
                "caption": fused_caption,
                "clothing": clothing,
                "color": color,
                "scene": scene,
                "embedding_id": meta.get("embedding_id", -1)
            })
            
        return output_results

if __name__ == "__main__":
    # CLI mode
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    parser = argparse.ArgumentParser(description="Query Advanced Fashion Retrieval.")
    parser.add_argument("query", type=str, help="Text search query")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results")
    
    args = parser.parse_args()
    
    try:
        retriever = FashionRetriever()
        results = retriever.retrieve(args.query, top_k=args.top_k)
        
        print("\n" + "="*70)
        print(f"SEARCH RESULTS FOR: '{args.query}'")
        print("="*70)
        for i, res in enumerate(results):
            print(f"\n{i+1}. Image: {res['image_path']}")
            print(f"   Weighted Similarity Score: {res['score']:.4f}")
            print(f"   Caption: {res['caption']}")
        print("="*70 + "\n")
    except Exception as e:
        logger.error(f"Search query failed: {e}", exc_info=True)
