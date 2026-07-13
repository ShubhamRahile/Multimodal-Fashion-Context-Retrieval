import os
import torch

class Config:
    """Centralized configuration for the Multimodal Fashion Retrieval system."""
    
    # Base paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Directory paths
    DATASET_DIR = os.path.join(BASE_DIR, "dataset", "images")
    EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
    OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
    
    # Output file paths
    METADATA_PATH = os.path.join(OUTPUTS_DIR, "metadata.json")
    FAISS_INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "faiss_index.index")
    EMBEDDINGS_RAW_PATH = os.path.join(EMBEDDINGS_DIR, "raw_embeddings.npy")
    
    # Model settings
    # OpenCLIP configuration
    CLIP_MODEL_NAME = "ViT-B-32"
    CLIP_PRETRAINED = "laion2b_s34b_b79k"
    
    # BLIP captioning model setting
    # We use Salesforce/blip-image-captioning-base for lightweight, fast execution on 4GB VRAM
    BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"
    
    # Device configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Hyperparameters
    BATCH_SIZE = 16
    FUSION_ALPHA = 0.6  # Fusion weight: Alpha * Image_Embedding + (1 - Alpha) * Caption_Embedding
    
    # Structured categories for zero-shot classification & metadata extraction
    CLOTHING_CATEGORIES = [
        "shirt", "t-shirt", "pants", "jeans", "shorts", "jacket", "coat", 
        "raincoat", "suit", "blazer", "dress", "skirt", "sweater", "hoodie", 
        "tie", "formal wear", "activewear", "casual wear"
    ]
    
    COLOR_CATEGORIES = [
        "red", "blue", "yellow", "green", "black", "white", "gray", "brown", 
        "pink", "purple", "orange", "beige", "navy", "multicolor"
    ]
    
    SCENE_CATEGORIES = [
        "modern office", "indoor office", "outdoor park", "park bench", 
        "city street", "street walk", "formal hall", "casual room", 
        "nature", "studio backdrop", "urban environment"
    ]

    @classmethod
    def create_dirs(cls) -> None:
        """Create necessary directories if they don't exist."""
        os.makedirs(cls.DATASET_DIR, exist_ok=True)
        os.makedirs(cls.EMBEDDINGS_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUTS_DIR, exist_ok=True)
