import os
import sys
import logging
from typing import List, Dict, Any, Tuple
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import Config

logger = logging.getLogger(__name__)

class RankingEngine:
    """Computes final multi-similarity scores and re-ranks retrieved candidates."""

    def __init__(self):
        """Initializes the re-ranking engine."""
        pass

    def rerank_results(self, query: str, query_embedding: np.ndarray, 
                       results: List[Tuple[float, Dict[str, Any]]]) -> List[Tuple[float, Dict[str, Any]]]:
        """Applies advanced multi-similarity re-ranking to Top-20 candidates.
        
        Formula:
          Final Score = 0.45 * ImageSim + 0.30 * CaptionSim + 0.15 * MetadataSim + 0.10 * SceneSim
        """
        query_lower = query.lower()
        reranked_results = []
        
        # Extracted query color keywords for color filtering/explainability
        query_colors = [c for c in Config.COLOR_CATEGORIES if c in query_lower]
        
        for score, meta in results:
            try:
                # Fallback design: check if raw embeddings are cached in metadata.json
                has_cached_embeddings = all(k in meta for k in ["image_embedding", "caption_embedding", "metadata_embedding", "scene_embedding"])
                
                if has_cached_embeddings:
                    # Convert cached lists of floats back to numpy arrays
                    E_image = np.array(meta["image_embedding"], dtype=np.float32)
                    E_caption = np.array(meta["caption_embedding"], dtype=np.float32)
                    E_metadata = np.array(meta["metadata_embedding"], dtype=np.float32)
                    E_scene = np.array(meta["scene_embedding"], dtype=np.float32)
                    
                    # Compute individual cosine similarities (dot products on normalized vectors)
                    sim_image = float(np.dot(query_embedding, E_image))
                    sim_caption = float(np.dot(query_embedding, E_caption))
                    sim_metadata = float(np.dot(query_embedding, E_metadata))
                    sim_scene = float(np.dot(query_embedding, E_scene))
                else:
                    # Fallback to base score and estimate components if index hasn't been regenerated yet
                    sim_image = score
                    sim_caption = score * 0.8
                    sim_metadata = score * 0.7
                    sim_scene = score * 0.6
                
                # Apply base weighted fusion score formula
                final_score = (
                    0.45 * sim_image + 
                    0.30 * sim_caption + 
                    0.15 * sim_metadata + 
                    0.10 * sim_scene
                )
                
                # Boost final score if dominant color matches the query color keyword (Color similarity re-ranking)
                detected_color = meta.get("detected_color", "").lower()
                color_match = False
                if detected_color and any(qc == detected_color for qc in query_colors):
                    final_score += 0.10  # Moderate boost for exact color match
                    color_match = True
                    
                # Cap score at 1.0
                final_score = min(final_score, 1.0)
                
                # Cache scores on the metadata dict for retrieval explainability (used in search.py)
                meta_copy = meta.copy()
                meta_copy["_sim_image"] = sim_image
                meta_copy["_sim_caption"] = sim_caption
                meta_copy["_sim_metadata"] = sim_metadata
                meta_copy["_sim_scene"] = sim_scene
                meta_copy["_color_match"] = color_match
                
                reranked_results.append((final_score, meta_copy))
                
            except Exception as e:
                logger.error(f"Error re-ranking candidate {meta.get('image_path')}: {e}")
                # Safe fallback
                reranked_results.append((score, meta))
                
        # Re-sort descending based on the final calculated score
        reranked_results.sort(key=lambda x: x[0], reverse=True)
        return reranked_results
