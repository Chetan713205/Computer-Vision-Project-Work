from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
from PIL import Image
import io
import asyncio
from typing import Dict, Any

from app.models.clothing_detector import ClothingDetector
from app.models.attribute_extractor import AttributeExtractor
from app.models.color_analyzer import ColorAnalyzer
from app.schemas.response import ClothingAnalysisResponse
from app.utils.image_processing import preprocess_image

app = FastAPI(title="Clothing Attribute Detection API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Initialize models (loaded once at startup)
clothing_detector = None
attribute_extractor = None
color_analyzer = None

@app.on_event("startup")
async def load_models():
    global clothing_detector, attribute_extractor, color_analyzer
    print("Loading models...")
    
    clothing_detector = ClothingDetector()
    attribute_extractor = AttributeExtractor()
    color_analyzer = ColorAnalyzer()
    
    print("Models loaded successfully!")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("frontend/index.html", "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(html)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Clothing Attribute Detection API is running"}

@app.post("/analyze", response_model=ClothingAnalysisResponse)
async def analyze_clothing(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and preprocess image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        processed_image = preprocess_image(image)
        
        # Run analysis in parallel
        detection_task = asyncio.create_task(
            clothing_detector.detect_clothing_items(processed_image)
        )
        attribute_task = asyncio.create_task(
            attribute_extractor.extract_attributes(processed_image)
        )
        color_task = asyncio.create_task(
            color_analyzer.analyze_colors(processed_image)
        )
        
        # Wait for all tasks to complete
        clothing_items, attributes, color_analysis = await asyncio.gather(
            detection_task, attribute_task, color_task
        )
        
        # Combine results
        result = {
            "status": "success",
            "clothing_items": clothing_items,
            "style_classification": attributes.get("style", "unknown"),
            "formality": attributes.get("formality", "unknown"), 
            "texture": attributes.get("texture", "unknown"),
            "dominant_colors": color_analysis["dominant_colors"],
            "color_distribution": color_analysis["color_distribution"],
            "detailed_attributes": attributes,
            "confidence_scores": {
                "overall": 0.85,
                "style": attributes.get("confidence", 0.8),
                "color": color_analysis.get("confidence", 0.9)
            }
        }
        
        return ClothingAnalysisResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
