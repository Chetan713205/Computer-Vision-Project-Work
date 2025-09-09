import torch
from sklearn.cluster import KMeans
import cv2
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image
import numpy as np
from typing import List, Dict, Any
import asyncio

class ClothingDetector:
    def __init__(self):
        self.model_name = "yainage90/fashion-object-detection"
        self.device = 'cpu'  # Force CPU usage
        self.processor = None
        self.model = None
        self._load_model()
        
    
    
    def _load_model(self):
        """Load the pre-trained fashion detection model"""
        try:
            print("Loading clothing detection model...")
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForObjectDetection.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            print("Clothing detection model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    async def detect_clothing_items(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect clothing items in the image"""
        try:
            # Run inference in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, self._run_detection, image)
            return results
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def _run_detection(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Run the actual detection"""
        with torch.no_grad():
            inputs = self.processor(images=[image], return_tensors="pt")
            outputs = self.model(**inputs.to(self.device))
            
            target_sizes = torch.tensor([[image.size[1], image.size[0]]])
            results = self.processor.post_process_object_detection(
                outputs, threshold=0.4, target_sizes=target_sizes
            )[0]
            
            items = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                items.append({
                    "item_type": self.model.config.id2label[label.item()],
                    "confidence": round(score.item(), 3),
                    "bounding_box": [round(i.item()) for i in box]
                })
            
            return items
