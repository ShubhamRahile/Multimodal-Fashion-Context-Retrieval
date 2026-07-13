import os
import shutil
import glob
import logging
from utils.config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def setup_project_dataset(num_samples: int = 1200) -> None:
    """Creates project directories and copies a subset of images from Dataset/train to dataset/images."""
    # Ensure directories exist
    Config.create_dirs()
    logger.info("Created output and target dataset directories.")

    # Check if target dataset already has images
    existing_images = glob.glob(os.path.join(Config.DATASET_DIR, "*.jpg"))
    if len(existing_images) >= num_samples:
        logger.info(f"Target dataset already has {len(existing_images)} images. Setup skipped.")
        return

    # Look for the source images in the project root's Dataset/train directory
    # The current working directory should contain Dataset/train/
    source_dir = os.path.join(Config.BASE_DIR, "Dataset", "train")
    if not os.path.exists(source_dir):
        # Fallback to check just Dataset/
        source_dir = os.path.join(Config.BASE_DIR, "Dataset")
        
    if not os.path.exists(source_dir):
        logger.error(f"Source dataset folder not found at '{source_dir}'. Cannot copy images.")
        return

    # List all JPG files in the source folder
    source_images = glob.glob(os.path.join(source_dir, "*.jpg")) + glob.glob(os.path.join(source_dir, "*.png"))
    if not source_images:
        # Check subdirectories of source_dir
        source_images = glob.glob(os.path.join(source_dir, "*", "*.jpg")) + glob.glob(os.path.join(source_dir, "*", "*.png"))

    if not source_images:
        logger.error(f"No images found in source folder '{source_dir}'.")
        return

    logger.info(f"Found {len(source_images)} source images. Copying {num_samples} samples...")

    # Copy files
    copied_count = 0
    for i, img_path in enumerate(source_images[:num_samples]):
        filename = os.path.basename(img_path)
        dest_path = os.path.join(Config.DATASET_DIR, filename)
        
        try:
            shutil.copy2(img_path, dest_path)
            copied_count += 1
        except Exception as e:
            logger.error(f"Failed to copy {filename}: {e}")

    logger.info(f"Successfully copied {copied_count} images to '{Config.DATASET_DIR}'.")

if __name__ == "__main__":
    setup_project_dataset()
