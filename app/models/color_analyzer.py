import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from typing import Dict, List, Any
import asyncio
import webcolors

class ColorAnalyzer:
    def __init__(self):
        self.color_names = {
            'red': [255, 0, 0],
            'green': [0, 255, 0], 
            'blue': [0, 0, 255],
            'yellow': [255, 255, 0],
            'orange': [255, 165, 0],
            'purple': [128, 0, 128],
            'pink': [255, 192, 203],
            'brown': [165, 42, 42],
            'black': [0, 0, 0],
            'white': [255, 255, 255],
            'gray': [128, 128, 128]
        }
    
    async def analyze_colors(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze colors in the clothing image"""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._extract_colors, image)
            return result
        except Exception as e:
            print(f"Color analysis error: {e}")
            return {"dominant_colors": [], "color_distribution": {}, "confidence": 0.0}
    
    def _extract_colors(self, image: Image.Image) -> Dict[str, Any]:
        """Extract dominant colors from image"""
        # Convert PIL image to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Resize image for faster processing
        height, width = opencv_image.shape[:2]
        if width > 300:
            scale = 300 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            opencv_image = cv2.resize(opencv_image, (new_width, new_height))
        
        # Reshape image for KMeans clustering
        data = opencv_image.reshape((-1, 3))
        data = np.float32(data)
        
        # Apply KMeans to find dominant colors
        k = 5  # Number of dominant colors to find
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert back to RGB and find closest color names
        centers = np.uint8(centers)
        dominant_colors = []
        color_distribution = {}
        
        # Count pixels for each cluster
        unique_labels, counts = np.unique(labels.flatten(), return_counts=True)
        total_pixels = len(labels)
        total_pixels = len(labels)
        
        # DEBUG: ensure labels shape matches centers
        #print(f"[DEBUG] unique_labels: {unique_labels}, counts: {counts}")
        for idx, (label, count) in enumerate(zip(unique_labels, counts)):
            # In OpenCV kmeans, label indices correspond to centers rows
            color_bgr = centers[label]
            color_rgb = [int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0])]  # BGR to RGB
            
            # Find closest named color
            color_name = self._get_closest_color_name(color_rgb)
            #print(f"[DEBUG] Cluster {idx}: RGB {color_rgb}, named as {color_name}, count {count}")
            percentage = (count / total_pixels) * 100
            
            dominant_colors.append({
                "color_name": color_name,
                "rgb": color_rgb,
                "hex": "#{:02x}{:02x}{:02x}".format(color_rgb[0], color_rgb[1], color_rgb[2]),
                "percentage": round(percentage, 2)
            })
            
            color_distribution[color_name] = round(percentage, 2)
        
        # Sort by percentage
        dominant_colors.sort(key=lambda x: x["percentage"], reverse=True)
        
        return {
            "dominant_colors": dominant_colors[:3],  # Top 3 colors
            "color_distribution": color_distribution,
            "confidence": 0.9
        }
    
    def _get_closest_color_name(self, rgb_color: List[int]) -> str:
        """Find the closest named color to the given RGB value"""
        # Convert RGB to HSV for hue-based matching
        import colorsys
        r, g, b = [c/255.0 for c in rgb_color]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        # Define target hues for named colors (approximate)
        hue_map = {
            'red': 0.0,
            'orange': 0.08,
            'yellow': 0.16,
            'green': 0.33,
            'blue': 0.61,
            'purple': 0.78,
            'pink': 0.92,
            'brown': 0.08,
            'gray': None,
            'black': None,
            'white': None
        }
        # If low saturation or extreme brightness, treat separately
        if v < 0.2:
            return 'black'
        if v > 0.9 and s < 0.15:
            return 'white'
        if s < 0.15 and 0.2 < v < 0.9:
            return 'gray'
        # Otherwise, find closest hue
        min_diff = 1.0
        closest = 'unknown'
        for name, target_h in hue_map.items():
            if target_h is None or name in ('black','white','gray'):
                continue
            diff = abs(h - target_h)
            diff = min(diff, 1 - diff)
            if diff < min_diff:
                min_diff = diff
                closest = name
        return closest
