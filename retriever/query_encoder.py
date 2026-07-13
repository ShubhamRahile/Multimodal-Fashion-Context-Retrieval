import os
import sys
import logging
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import Config
from indexer.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)

class QueryEncoder:
    """Encodes natural language queries into normalized OpenCLIP embeddings."""

    def __init__(self, feature_extractor: FeatureExtractor = None):
        """Initializes the QueryEncoder with an optional pre-loaded FeatureExtractor."""
        if feature_extractor is not None:
            self.extractor = feature_extractor
        else:
            logger.info("Initializing standalone FeatureExtractor for QueryEncoder...")
            self.extractor = FeatureExtractor()

    def encode_query(self, query: str) -> np.ndarray:
        """Encodes the text query and returns a normalized embedding vector."""
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")
            
        logger.info(f"Encoding text query: '{query}'")
        
        # Get normalized embedding from OpenCLIP
        embedding = self.extractor.get_text_embedding(query)
        
        # L2 normalization to ensure inner product yields cosine similarity
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
