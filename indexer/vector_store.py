import os
import sys
import json
import logging
from typing import List, Dict, Any, Tuple
import faiss
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import Config

logger = logging.getLogger(__name__)

class VectorStore:
    """Manages the FAISS index and stores image metadata in JSON format."""

    def __init__(self, index_path: str = Config.FAISS_INDEX_PATH, 
                 metadata_path: str = Config.METADATA_PATH):
        """Initializes the vector store paths and loads existing index/metadata if available."""
        self.index_path = index_path
        self.metadata_path = metadata_path
        
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        
        self.load()

    def _init_index(self, dimension: int) -> None:
        """Initializes a new FAISS Inner Product index (for Cosine Similarity)."""
        logger.info(f"Initializing a new FAISS IndexFlatIP with dimension {dimension}...")
        self.index = faiss.IndexFlatIP(dimension)

    def add_vectors(self, embeddings: np.ndarray, metadata_list: List[Dict[str, Any]]) -> None:
        """Adds embeddings and corresponding metadata incrementally.
        
        Embeddings must be L2 normalized before calling this method to ensure IndexFlatIP
        computes Cosine Similarity.
        """
        if len(embeddings) != len(metadata_list):
            raise ValueError(f"Mismatch: Got {len(embeddings)} embeddings and {len(metadata_list)} metadata records.")
            
        if len(embeddings) == 0:
            return
            
        dimension = embeddings.shape[1]
        
        if self.index is None:
            self._init_index(dimension)
            
        # Ensure embeddings are float32 (FAISS requirement)
        embeddings_f32 = embeddings.astype(np.float32)
        
        # Add to FAISS index
        start_id = self.index.ntotal
        self.index.add(embeddings_f32)
        
        # Append to metadata and assign Embedding IDs
        for i, meta in enumerate(metadata_list):
            meta_copy = meta.copy()
            meta_copy["embedding_id"] = start_id + i
            self.metadata.append(meta_copy)
            
        logger.info(f"Added {len(embeddings)} vectors to the index. Total vectors: {self.index.ntotal}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        """Searches the FAISS index for the top K closest vectors.
        
        Query embedding must be L2 normalized.
        Returns a list of tuples containing (similarity_score, metadata).
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Search called on an empty vector store index.")
            return []
            
        # Reshape to 2D array if 1D
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        query_embedding_f32 = query_embedding.astype(np.float32)
        
        # Search index
        scores, indices = self.index.search(query_embedding_f32, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue
            
            # Since IndexFlatIP is inner product, and vectors are normalized, score is directly Cosine Similarity.
            # Convert score to regular float for JSON serialization
            results.append((float(score), self.metadata[idx]))
            
        return results

    def save(self) -> None:
        """Saves both the FAISS index and metadata file to disk."""
        if self.index is None:
            logger.warning("No index available to save.")
            return
            
        try:
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, self.index_path)
            
            # Save metadata
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=4, ensure_ascii=False)
                
            logger.info(f"Saved FAISS index to '{self.index_path}' and metadata to '{self.metadata_path}'.")
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
            raise

    def load(self) -> None:
        """Loads index and metadata from disk if they exist."""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                # Load FAISS index
                self.index = faiss.read_index(self.index_path)
                
                # Load metadata
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                    
                logger.info(f"Loaded existing index with {self.index.ntotal} vectors from '{self.index_path}'.")
            except Exception as e:
                logger.error(f"Failed to load existing index/metadata: {e}. Starting fresh.")
                self.index = None
                self.metadata = []
        else:
            logger.info("No pre-existing FAISS index found. Ready to initialize on indexing.")
            self.index = None
            self.metadata = []

    def get_total_vectors(self) -> int:
        """Returns the number of vectors stored in the index."""
        return self.index.ntotal if self.index is not None else 0
