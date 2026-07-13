import os
import sys
import logging
from typing import Dict, Any, List
import numpy as np
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import Config
from indexer.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)

class MetadataGenerator:
    """Extracts detailed visual metadata using OpenCV color clustering and OpenCLIP zero-shot classification."""

    def __init__(self, feature_extractor: FeatureExtractor):
        """Initializes templates and pre-encodes prompt categories."""
        self.extractor = feature_extractor
        
        # Advanced classifications categories
        self.sleeve_categories = ["long sleeve", "short sleeve", "sleeveless"]
        self.gender_categories = ["men's", "women's", "unisex"]
        self.environment_categories = ["indoor", "outdoor"]
        self.style_categories = ["casual", "formal", "business casual", "streetwear", "sporty", "minimalist"]
        self.scene_categories = ["office", "street", "park", "home", "cafe", "mall"]
        self.activity_categories = ["walking", "sitting", "standing", "running"]

        # Pre-encode templates for zero-shot classification
        logger.info("Pre-encoding advanced metadata categories...")
        self.clothing_embeddings = self._encode_templates(Config.CLOTHING_CATEGORIES, "a photo of a person wearing {}")
        self.sleeve_embeddings = self._encode_templates(self.sleeve_categories, "clothing with {}s")
        self.gender_embeddings = self._encode_templates(self.gender_categories, "{} fashion wear")
        self.environment_embeddings = self._encode_templates(self.environment_categories, "an {} setting")
        self.style_embeddings = self._encode_templates(self.style_categories, "{} style clothing")
        self.scene_embeddings = self._encode_templates(self.scene_categories, "a photo taken inside a {}")
        self.activity_embeddings = self._encode_templates(self.activity_categories, "a person {}")

        # Color standard RGB map for dominant color name calculation
        self.color_rgb_map = {
            "black": (15, 15, 15),
            "white": (240, 240, 240),
            "gray": (128, 128, 128),
            "red": (220, 20, 60),
            "blue": (30, 144, 255),
            "yellow": (255, 215, 0),
            "green": (34, 139, 34),
            "pink": (255, 105, 180),
            "purple": (138, 43, 226),
            "orange": (255, 69, 0),
            "brown": (139, 69, 19),
            "beige": (245, 245, 220),
            "navy": (0, 0, 128)
        }

    def _encode_templates(self, categories: List[str], template: str) -> np.ndarray:
        """Encodes list of categories with standard template, managing recursion state."""
        self.extractor.is_encoding_templates = True
        embeddings = []
        try:
            for cat in categories:
                prompt = template.format(cat)
                emb = self.extractor.get_text_embedding(prompt)
                embeddings.append(emb)
        finally:
            self.extractor.is_encoding_templates = False
            
        return np.stack(embeddings, axis=0)

    def extract_dominant_color_opencv(self, image_path: str) -> str:
        """Extracts the dominant garment color using OpenCV K-Means clustering."""
        abs_path = image_path
        if not os.path.isabs(abs_path):
            abs_path = os.path.join(Config.BASE_DIR, image_path)

        if not os.path.exists(abs_path):
            return "multicolor"

        try:
            img = cv2.imread(abs_path)
            if img is None:
                return "multicolor"
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Crop center to focus on garment
            h, w, _ = img.shape
            crop_h, crop_w = int(h * 0.4), int(w * 0.4)
            start_y, start_x = int(h * 0.3), int(w * 0.3)
            cropped = img[start_y : start_y + crop_h, start_x : start_x + crop_w]
            
            resized = cv2.resize(cropped, (50, 50))
            pixels = resized.reshape(-1, 3).astype(np.float32)
            
            k = 3
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            flags = cv2.KMEANS_RANDOM_CENTERS
            _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, flags)
            
            counts = np.bincount(labels.flatten())
            sorted_indices = np.argsort(counts)[::-1]
            
            dominant_rgb = centers[sorted_indices[0]]
            
            closest_color = "gray"
            min_distance = float("inf")
            for color_name, target_rgb in self.color_rgb_map.items():
                distance = np.linalg.norm(np.array(dominant_rgb) - np.array(target_rgb))
                if distance < min_distance:
                    min_distance = distance
                    closest_color = color_name
                    
            return closest_color
        except Exception as e:
            logger.warning(f"OpenCV dominant color extraction failed for {image_path}: {e}. Defaulting to zero-shot color.")
            return "multicolor"

    def extract_metadata(self, image_path: str, image_embedding: np.ndarray, caption: str) -> Dict[str, Any]:
        """Extracts complete fashion metadata dict and formats summary details."""
        # 1. Zero-shot CLIP classification
        clothing_idx = self._classify(image_embedding, self.clothing_embeddings)
        clothing = Config.CLOTHING_CATEGORIES[clothing_idx]
        
        sleeve_idx = self._classify(image_embedding, self.sleeve_embeddings)
        sleeve = self.sleeve_categories[sleeve_idx]
        
        gender_idx = self._classify(image_embedding, self.gender_embeddings)
        gender = self.gender_categories[gender_idx]
        
        env_idx = self._classify(image_embedding, self.environment_embeddings)
        environment = self.environment_categories[env_idx]
        
        style_idx = self._classify(image_embedding, self.style_embeddings)
        style = self.style_categories[style_idx]
        
        scene_idx = self._classify(image_embedding, self.scene_embeddings)
        scene = self.scene_categories[scene_idx]
        
        activity_idx = self._classify(image_embedding, self.activity_embeddings)
        activity = self.activity_categories[activity_idx]

        # 2. OpenCV dominant color matching
        opencv_color = self.extract_dominant_color_opencv(image_path)
        
        final_color = opencv_color
        if opencv_color == "multicolor":
            color_embeddings = self._encode_templates(Config.COLOR_CATEGORIES, "a photo of {} clothing")
            color_idx = self._classify(image_embedding, color_embeddings)
            final_color = Config.COLOR_CATEGORIES[color_idx]

        # 3. Format structured metadata text
        metadata_summary = (
            f"clothing: {clothing}, color: {final_color}, sleeve: {sleeve}, "
            f"gender: {gender}, environment: {environment}, style: {style}, "
            f"location: {scene}, activity: {activity}"
        )
        
        # Build structured JSON metadata with raw embeddings lists
        return {
            "image_path": image_path,
            "caption": caption,
            "detected_clothing": clothing,
            "detected_color": final_color,
            "sleeve": sleeve,
            "gender": gender,
            "environment": environment,
            "style": style,
            "scene": scene,
            "activity": activity,
            "metadata_summary_text": metadata_summary,
            "image_embedding": image_embedding.tolist(),
            "caption_embedding": self.extractor.last_caption_emb.tolist() if self.extractor.last_caption_emb is not None else [],
            "metadata_embedding": self.extractor.last_metadata_emb.tolist() if self.extractor.last_metadata_emb is not None else [],
            "scene_embedding": self.extractor.last_scene_emb.tolist() if self.extractor.last_scene_emb is not None else []
        }

    def _classify(self, embedding: np.ndarray, prompt_embeddings: np.ndarray) -> int:
        """Computes cosine similarity between image embedding and template embeddings."""
        similarities = np.dot(prompt_embeddings, embedding)
        return int(np.argmax(similarities))
