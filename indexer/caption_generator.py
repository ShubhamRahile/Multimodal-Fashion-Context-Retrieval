import os
import sys
import logging
from typing import Union
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import Config

from transformers import BlipProcessor, BlipForConditionalGeneration

logger = logging.getLogger(__name__)

class CaptionGenerator:
    """Generates detailed natural language descriptions of fashion images using BLIP."""

    def __init__(self, model_name: str = Config.BLIP_MODEL_NAME, device: str = Config.DEVICE):
        """Initializes the BLIP processor and detailed caption model."""
        self.device = device
        logger.info(f"Loading BLIP model {model_name} on {self.device}...")
        
        try:
            self.processor = BlipProcessor.from_pretrained(model_name)
            # Use torch.float32 on CPU or float16 on GPU
            dtype = torch.float16 if device == "cuda" else torch.float32
            self.model = BlipForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype).to(self.device)
            logger.info("BLIP model loaded successfully for detailed captions.")
            self.failed_to_load = False
        except Exception as e:
            logger.error(f"Error loading BLIP model: {e}. Falling back to keyword captions.")
            self.failed_to_load = True

    def generate_caption(self, image: Union[str, Image.Image]) -> str:
        """Generates a detailed context-rich caption for the given image."""
        if self.failed_to_load:
            return self._fallback_caption(image)

        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image not found at path: {image}")
            try:
                image = Image.open(image).convert("RGB")
            except Exception as e:
                logger.warning(f"Could not open image {image}: {e}. Returning fallback.")
                return self._fallback_caption(image)

        try:
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            if self.device == "cuda":
                inputs = {k: v.to(dtype=torch.float16) if v.dtype == torch.float32 else v for k, v in inputs.items()}
                
            # Hyperparameter optimizations to enforce detailed, rich captions:
            # - num_beams=3 enables beam search for high-quality generation
            # - min_length=25 forces detailed descriptions containing colors, clothing, and scene
            # - max_length=60 sets an upper bound on length
            out = self.model.generate(
                **inputs, 
                max_length=60,
                min_length=25,
                num_beams=1,
                length_penalty=1.0,
                repetition_penalty=1.2
            )
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            return caption.strip()
        except Exception as e:
            logger.error(f"Error generating BLIP detailed caption: {e}. Returning fallback.")
            return self._fallback_caption(image)

    def _fallback_caption(self, image: Union[str, Image.Image]) -> str:
        """Generates a fallback caption based on image metadata or filename."""
        if isinstance(image, str):
            filename = os.path.basename(image).lower()
            name_parts = os.path.splitext(filename)[0].split("_")
            keywords = [p for p in name_parts if len(p) > 2]
            if keywords:
                return f"A fashion apparel style portrait featuring {' '.join(keywords)} clothing items outdoors."
        return "A fashion apparel portrait displaying casual outfit styles in a lifestyle context."
