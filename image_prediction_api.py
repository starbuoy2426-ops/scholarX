from io import BytesIO

import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

ALLOWED_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(
    title="Image Prediction API",
    version="1.0.0",
    description="Professional FastAPI service for image classification using a pretrained ResNet50 model.",
)

weights = ResNet50_Weights.DEFAULT

model = resnet50(weights=weights).to(device)
model.eval()

categories = weights.meta["categories"]

image_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

class PredictionResponse(BaseModel):
    filename: str
    prediction: str
    confidence: float

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid image.",
        ) from exc

    return image_transform(image).unsqueeze(0).to(device)

@app.get("/")
def home():
    return {
        "message": "Welcome to Image Prediction API",
        "documentation": "/docs",
        "health": "/health",
        "prediction_endpoint": "/predict",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "ResNet50",
        "device": str(device),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # Validate content type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only JPEG, JPG, PNG, WEBP, and BMP images are supported.",
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Image exceeds the maximum allowed size of 5 MB.",
        )

    tensor = preprocess_image(image_bytes)

    try:
        with torch.inference_mode():
            outputs = model(tensor)
            probabilities = torch.softmax(outputs[0], dim=0)
            confidence, class_index = torch.max(probabilities, dim=0)

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(exc)}",
        ) from exc

    prediction = categories[class_index.item()]

    return PredictionResponse(
        filename=file.filename or "upload",
        prediction=prediction,
        confidence=round(confidence.item(), 4),
    )

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8011,
        log_level="info",
    )