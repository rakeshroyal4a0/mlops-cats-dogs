import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File
from torchvision import transforms
from PIL import Image
import io
import os

from src.model import SimpleCNN

# Initialize FastAPI app
app = FastAPI()

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = SimpleCNN().to(DEVICE)
import os

# Load model safely
model = SimpleCNN().to(DEVICE)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "model.pt")

model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()

# Image transform (NO augmentation for inference)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class_names = ["Cat", "Dog"]

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "Model is running"}

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    return {
        "prediction": class_names[predicted.item()],
        "confidence": float(confidence.item())
    }
