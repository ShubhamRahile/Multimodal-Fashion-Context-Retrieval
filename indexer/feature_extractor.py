import os
import sys
import logging
import inspect
from typing import List, Union, Tuple
import torch
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import Config

import open_clip

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Handles image and text embedding extraction using OpenCLIP and manages fusion states."""

    def __init__(self, model_name: str = Config.CLIP_MODEL_NAME, 
                 pretrained: str = Config.CLIP_PRETRAINED, 
                 device: str = Config.DEVICE):
        """Initializes model, tokenizer, and state variables."""
        self.device = device
        logger.info(f"Loading OpenCLIP model {model_name} (pretrained={pretrained}) on {self.device}...")
        
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, 
                pretrained=pretrained, 
                device=self.device
            )
            self.model.eval()
            self.tokenizer = open_clip.get_tokenizer(model_name)
            logger.info("OpenCLIP model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading OpenCLIP model: {e}")
            raise
            
        # Overwrite FUSION_ALPHA dynamically to 0.45 to support the weighted fusion formula
        Config.FUSION_ALPHA = 0.45
        logger.info("Config.FUSION_ALPHA dynamically set to 0.45 for 4-way fusion.")

        # Pipeline state caches
        self.last_img_path = None
        self.last_img_emb = None
        self.last_caption_emb = None
        self.last_metadata_emb = None
        self.last_scene_emb = None
        self.meta_gen = None
        
        # Recursion control flag
        self.is_encoding_templates = False

    @torch.no_grad()
    def get_image_embedding(self, image: Union[str, Image.Image]) -> np.ndarray:
        """Extracts and normalizes raw image embedding."""
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image not found at path: {image}")
            img_path = image
            image = Image.open(image).convert("RGB")
        else:
            img_path = "in_memory"

        # Embed image
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        image_features = self.model.encode_image(image_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        img_emb = image_features.cpu().numpy()[0]
        
        self.last_img_path = img_path
        self.last_img_emb = img_emb
        
        return img_emb

    def _get_raw_text_embedding(self, text: str) -> np.ndarray:
        """Retrieves raw normalized text embedding from OpenCLIP."""
        tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy()[0]

    def get_text_embedding(self, text: str) -> np.ndarray:
        """Fetches text embedding, applying 4-way weighted fusion only during indexing calls."""
        stack_filenames = [frame.filename for frame in inspect.stack()]
        is_indexing_call = any("image_indexer.py" in filename for filename in stack_filenames)
        
        # Bypasses fusion recursion when encoding category prompts
        if is_indexing_call and self.last_img_emb is not None and not self.is_encoding_templates:
            try:
                from indexer.metadata_generator import MetadataGenerator
                if self.meta_gen is None:
                    self.meta_gen = MetadataGenerator(self)
                
                # Retrieve zero-shot classification structures
                rel_path = os.path.relpath(self.last_img_path, start=Config.BASE_DIR).replace("\\", "/")
                metadata = self.meta_gen.extract_metadata(rel_path, self.last_img_emb, text)
                
                metadata_text = metadata["metadata_summary_text"]
                scene_text = f"a photo taken in a {metadata['scene']}"
                
                # Retrieve raw embeddings for fusion components
                E_caption = self._get_raw_text_embedding(text)
                E_metadata = self._get_raw_text_embedding(metadata_text)
                E_scene = self._get_raw_text_embedding(scene_text)
                
                # Cache these components for metadata.json mapping
                self.last_caption_emb = E_caption
                self.last_metadata_emb = E_metadata
                self.last_scene_emb = E_scene
                
                # Blend: Fused = 0.45 * Image + 0.55 * CombinedText
                # CombinedText = (0.30 * Caption + 0.15 * Metadata + 0.10 * Scene) / 0.55
                combined_text_emb = (0.30 * E_caption + 0.15 * E_metadata + 0.10 * E_scene) / 0.55
                return combined_text_emb
                
            except Exception as e:
                logger.error(f"Error inside multi-vector fusion: {e}. Falling back to raw text embedding.", exc_info=True)
                return self._get_raw_text_embedding(text)
        else:
            # Standalone query encoding (no fusion)
            return self._get_raw_text_embedding(text)

    def get_embedding_dimension(self) -> int:
        """Returns the visual feature dimension of the model."""
        if hasattr(self.model, "visual") and hasattr(self.model.visual, "output_dim"):
            return self.model.visual.output_dim
        dummy_input = torch.zeros(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            dummy_out = self.model.encode_image(dummy_input)
        return dummy_out.shape[-1]
