import os
import sys
import glob
import logging
import numpy as np
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "indexer.log"), encoding="utf-8")
    ]
)
logger = logging.getLogger("indexer")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import Config
from utils.setup_dataset import setup_project_dataset
from indexer.feature_extractor import FeatureExtractor
from indexer.caption_generator import CaptionGenerator
from indexer.metadata_generator import MetadataGenerator
from indexer.vector_store import VectorStore

def run_indexing_pipeline(num_images_to_index: int = 600) -> None:
    """Orchestrates the entire fashion image indexing pipeline."""
    logger.info("Starting Multimodal Fashion Indexing Pipeline...")
    
    # Step 1: Initialize folders and copy dataset
    setup_project_dataset(num_samples=num_images_to_index)
    
    # Step 2: Initialize components
    logger.info("Initializing ML components...")
    extractor = FeatureExtractor()
    captioner = CaptionGenerator()
    meta_gen = MetadataGenerator(extractor)
    vector_db = VectorStore()
    
    # Find all images in dataset/images
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(Config.DATASET_DIR, ext)))
        
    image_paths = sorted(image_paths)
    total_images = len(image_paths)
    
    if total_images == 0:
        logger.error(f"No images found in dataset folder '{Config.DATASET_DIR}'. Exiting indexing.")
        return
        
    logger.info(f"Found {total_images} images in target folder.")
    
    # Step 3: Determine which images need indexing (Incremental Indexing)
    indexed_paths = {meta["image_path"] for meta in vector_db.metadata}
    new_image_paths = [p for p in image_paths if p not in indexed_paths]
    
    if len(new_image_paths) == 0:
        logger.info("All images are already indexed. No new indexing required.")
        return
        
    logger.info(f"{len(new_image_paths)} out of {total_images} images are new and will be indexed.")
    
    # Process new images
    batch_embeddings = []
    batch_metadata = []
    
    save_every = 50  # Save progress every 50 images to prevent loss
    
    for i, path in enumerate(tqdm(new_image_paths, desc="Indexing Images")):
        try:
            # 1. Open image
            relative_path = os.path.relpath(path, start=Config.BASE_DIR).replace("\\", "/")
            
            # 2. Extract CLIP Image Embedding
            img_emb = extractor.get_image_embedding(path)
            
            # 3. Generate BLIP Caption
            caption = captioner.generate_caption(path)
            
            # 4. Extract CLIP Text Embedding from Caption
            caption_emb = extractor.get_text_embedding(caption)
            
            # 5. Extract Structured Metadata (Clothing, Color, Scene)
            metadata = meta_gen.extract_metadata(relative_path, img_emb, caption)
            
            # 6. Fuse Image + Caption Embeddings (Visual-Semantic Embedding Fusion)
            # Both are normalized, so their weighted sum can be normalized again
            fused_emb = Config.FUSION_ALPHA * img_emb + (1 - Config.FUSION_ALPHA) * caption_emb
            fused_normalized_emb = fused_emb / np.linalg.norm(fused_emb)
            
            # Append to batches
            batch_embeddings.append(fused_normalized_emb)
            batch_metadata.append(metadata)
            
            # Save progress periodically
            if len(batch_embeddings) >= save_every or i == len(new_image_paths) - 1:
                # Add to vector store
                vector_db.add_vectors(np.array(batch_embeddings), batch_metadata)
                vector_db.save()
                
                # Reset batches
                batch_embeddings = []
                batch_metadata = []
                
        except Exception as e:
            logger.error(f"Failed to index image '{path}': {e}", exc_info=True)
            
    # Step 7: Save raw dense embeddings array as npy (for offline analysis)
    try:
        if vector_db.get_total_vectors() > 0:
            # We can recreate the matrix of all vectors in FAISS
            dimension = vector_db.index.d
            total_vecs = vector_db.index.ntotal
            raw_matrix = np.zeros((total_vecs, dimension), dtype=np.float32)
            
            # Extract reconstruct vectors from index
            for idx in range(total_vecs):
                raw_matrix[idx] = vector_db.index.reconstruct(idx)
                
            np.save(Config.EMBEDDINGS_RAW_PATH, raw_matrix)
            logger.info(f"Raw fused embeddings saved to '{Config.EMBEDDINGS_RAW_PATH}'.")
    except Exception as e:
        logger.error(f"Error saving raw numpy embeddings: {e}")
        
    logger.info("Multimodal Fashion Indexing Pipeline completed successfully.")

if __name__ == "__main__":
    # Index 600 images (satisfies GLANCE assignment requirement)
    run_indexing_pipeline(num_images_to_index=600)
