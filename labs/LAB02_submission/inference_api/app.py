"""
Congo Returns — Inference API

A FastAPI service that classifies product images using MobileNetV2.
Designed to run in a Docker container with a mounted /logs volume.

The API accepts image uploads and returns classifications from ImageNet's
1000 classes. In production, Congo would map these to their routing
categories (Electronics, Clothing, etc.).
"""

import json
import os
from datetime import datetime
from io import BytesIO
from pathlib import Path

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from torchvision import models, transforms

# ============================================================================
# Application Setup
# ============================================================================

app = FastAPI(
    title="Congo Returns - Inference API",
    description="Product classification service using MobileNetV2",
    version="1.0.0"
)

# Path where classification logs are written
# This is an in-container path; for it to persist, it should be bound to a host directory
LOG_PATH = Path("/logs/classifications.jsonl")

# ============================================================================
# Model Loading
# ============================================================================

# Load MobileNetV2 with pretrained ImageNet weights
# This happens once at startup to avoid reloading on every request
print("Loading MobileNetV2 model...")
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.eval()  # Set to evaluation mode (disables dropout, etc.)
print("Model loaded successfully!")

# Load ImageNet class labels
# This file must exist in the image at runtime; it maps indices (0-999) to human-readable class names
CLASSES_FILE = Path(__file__).parent / "imagenet_classes.json"
with open(CLASSES_FILE, 'r') as f:
    imagenet_classes = json.load(f)

# ============================================================================
# Image Preprocessing
# ============================================================================

# Standard ImageNet preprocessing pipeline
# MobileNetV2 expects images normalized with these specific values
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def classify_image(image: Image.Image) -> dict:
    """
    Classify a single image using MobileNetV2.
    
    Args:
        image: PIL Image object
        
    Returns:
        Dictionary with top prediction and confidence scores
    """
    # Preprocess the image
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
    
    # Run inference (no gradients needed)
    with torch.no_grad():
        output = model(input_batch)
    
    # Convert to probabilities
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    # Get top 5 predictions
    top5_prob, top5_idx = torch.topk(probabilities, 5)
    
    results = []
    for i in range(5):
        idx = top5_idx[i].item()
        results.append({
            "class": imagenet_classes[str(idx)],
            "confidence": round(top5_prob[i].item() * 100, 2)
        })
    
    return {
        "top_prediction": results[0]["class"],
        "confidence": results[0]["confidence"],
        "top_5": results
    }


# ============================================================================
# Logging
# ============================================================================

def log_classification(filename: str, result: dict, metadata: dict = None):
    """
    Append a classification result to the JSONL log file.
    
    Each line in the log is a JSON object with:
    - timestamp: When the classification occurred
    - filename: Original filename of the image
    - customer_id: Extracted from filename (if available)
    - product_id: Extracted from filename (if available)
    - top_prediction: The most likely class
    - confidence: Confidence score (0-100)
    """
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "filename": filename,
        "customer_id": metadata.get("customer_id") if metadata else None,
        "product_id": metadata.get("product_id") if metadata else None,
        "top_prediction": result["top_prediction"],
        "confidence": result["confidence"]
    }
    
    # Ensure the logs directory exists
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Append to the log file (one JSON object per line)
    with open(LOG_PATH, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
def root():
    """Root endpoint with API info."""
    return {
        "service": "Congo Returns - Inference API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Classify a product image",
            "/docs": "GET - Swagger UI documentation"
        }
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    customer_id: str = None,
    product_id: str = None
):
    """
    Classify a product image.
    
    Args:
        file: Image file (JPEG, PNG)
        customer_id: Optional customer ID for logging
        product_id: Optional product ID for logging
        
    Returns:
        Classification result with top prediction and confidence
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail=f"File must be an image. Got: {file.content_type}"
        )
    
    try:
        # Read and decode the image
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        
        # Convert to RGB if necessary (handles grayscale, RGBA, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Run classification
        result = classify_image(image)
        
        # Log the result
        metadata = {"customer_id": customer_id, "product_id": product_id}
        log_classification(file.filename, result, metadata)
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


# ============================================================================
# TODO: Add /health and /stats endpoints (Part 4)
# ============================================================================
# Your code here!
# 
# /health should return: {"status": "healthy", "model_loaded": true}
#
# /stats should read from the log file and return statistics like:
# - Total items processed
# - Breakdown by classification
# - Average confidence score


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) # The uvicorn server listens on port 8000 and binds to all interfaces
