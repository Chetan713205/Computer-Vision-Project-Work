from PIL import Image
import numpy as np
from typing import Tuple

def preprocess_image(image: Image.Image, target_size: Tuple[int, int] = (512, 512)) -> Image.Image:
    """Preprocess image for model inference"""
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize while maintaining aspect ratio
    image.thumbnail(target_size, Image.LANCZOS)
    
    # Create a new image with the target size and paste the resized image
    new_image = Image.new('RGB', target_size, (255, 255, 255))
    paste_position = ((target_size[0] - image.width) // 2, 
                     (target_size[1] - image.height) // 2)
    new_image.paste(image, paste_position)
    
    return new_image

def enhance_image_quality(image: Image.Image) -> Image.Image:
    """Enhance image quality for better analysis"""
    import cv2
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Apply slight gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(img_array, (3, 3), 0)
    
    # Enhance contrast
    enhanced = cv2.addWeighted(img_array, 1.5, blurred, -0.5, 0)
    
    return Image.fromarray(enhanced)
