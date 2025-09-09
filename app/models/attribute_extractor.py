import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import re
import asyncio
from typing import Dict, Any

class AttributeExtractor:
    def __init__(self):
        self.model_name = "Salesforce/blip-image-captioning-base"
        self.processor = None
        self.model = None
        self._load_model()
        
        # Define attribute patterns for text analysis
        self.style_patterns = {
            "formal": ["suit", "blazer", "dress shirt", "tie", "formal", "business", "elegant"],
            "casual": ["t-shirt", "jeans", "sneakers", "hoodie", "casual", "relaxed", "comfortable", "leggings"],
            "sports": ["athletic", "sports", "gym", "workout", "running", "training"]
        }
        
        self.texture_patterns = {
            "cotton": ["cotton", "soft", "comfortable"],
            "denim": ["denim", "jeans", "rugged"],
            "silk": ["silk", "smooth", "shiny", "lustrous", "leggings", "velvet"],
            "wool": ["wool", "warm", "thick"],
            "leather": ["leather", "tough", "durable"],
            "synthetic": ["polyester", "synthetic", "artificial"]
        }
    
    def _load_model(self):
        """Load the BLIP model for image captioning"""
        try:
            print("Loading BLIP model for attribute extraction...")
            self.processor = BlipProcessor.from_pretrained(self.model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(self.model_name)
            self.model.eval()
            print("BLIP model loaded successfully!")
        except Exception as e:
            print(f"Error loading BLIP model: {e}")
            raise
    
    async def extract_attributes(self, image: Image.Image) -> Dict[str, Any]:
        """Extract clothing attributes from image"""
        try:
            loop = asyncio.get_event_loop()
            
            # Generate multiple captions with different prompts
            tasks = [
                loop.run_in_executor(None, self._generate_caption, image, "a photo of"),
                loop.run_in_executor(None, self._generate_caption, image, "clothing style:"),
                loop.run_in_executor(None, self._generate_caption, image, "fabric texture:")
            ]
            
            captions = await asyncio.gather(*tasks)
            
            # Analyze captions to extract attributes
            attributes = self._analyze_captions(captions)
            return attributes
            
        except Exception as e:
            print(f"Attribute extraction error: {e}")
            return {"style": "unknown", "formality": "unknown", "texture": "unknown"}
    
    def _generate_caption(self, image: Image.Image, prompt: str = "") -> str:
        """Generate caption for the image"""
        try:
            if prompt:
                inputs = self.processor(image, prompt, return_tensors="pt")
            else:
                inputs = self.processor(image, return_tensors="pt")
            
            with torch.no_grad():
                out = self.model.generate(**inputs, max_length=50, num_beams=4)
                caption = self.processor.decode(out[0], skip_special_tokens=True)
            
            return caption.lower()
        except Exception as e:
            print(f"Caption generation error: {e}")
            return ""
    
    def _analyze_captions(self, captions: list) -> Dict[str, Any]:
        """Analyze captions to extract structured attributes"""
        combined_text = " ".join(captions).lower()
        
        # Determine style/formality
        formal_score = sum(1 for word in self.style_patterns["formal"] if word in combined_text)
        casual_score = sum(1 for word in self.style_patterns["casual"] if word in combined_text)
        sports_score = sum(1 for word in self.style_patterns["sports"] if word in combined_text)
        
        if formal_score > casual_score and formal_score > sports_score:
            style = "formal"
            formality = "formal"
        elif sports_score > casual_score:
            style = "athletic"
            formality = "casual"
        else:
            style = "casual"
            formality = "casual"
        
        # Determine texture
        texture_scores = {}
        for texture, patterns in self.texture_patterns.items():
            texture_scores[texture] = sum(1 for word in patterns if word in combined_text)
        
        detected_texture = max(texture_scores, key=texture_scores.get) if max(texture_scores.values()) > 0 else "unknown"
        
        return {
            "style": style,
            "formality": formality,
            "texture": detected_texture,
            "confidence": 0.8,
            "raw_captions": captions,
            "detected_keywords": [word for word in combined_text.split() if any(word in patterns for patterns in self.style_patterns.values())]
        }
