import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image

# ----------------------------
# 1️⃣ CONFIG
# ----------------------------

DATA_DIR = "data/raw"
MODEL_SAVE_PATH = "model.pth"
BATCH_SIZE = 32
EPOCHS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)

# ----------------------------
# 2️⃣ FIND PetImages FOLDER AUTOMATICALLY
# ----------------------------

pet_images_path = None

for root, dirs, files in os.walk(DATA_DIR):
    if "PetImages" in dirs:
        pet_images_path = os.path.join(root, "PetImages")
        break

if pet_images_path is None:
    raise Exception("❌ PetImages folder not found inside data/raw")

print("✅ Dataset found at:", pet_images_path)

# ----------------------------
# 3️⃣ CLEAN CORRUPTED IMAGES
# ----------------------------

for category in ["Cat", "Dog"]:
    folder_path = os.path.join(pet_images_path, category)

    if not os.path.exists(folder_path):
        continue

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        try:
            img = Image.open(file_path)
            img.verify()
        except Exception:
            print("Removing corrupted file:", file_path)
            os.remove(file_path)

print("✅ Data cleaning completed")

# ----------------------------
# 4️⃣ TRANSFORMS
# ----------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ----------------------------
# 5️⃣ LOAD DATASET
# ----------------------------

dataset = datasets.ImageFolder(pet_images_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print("✅ Dataset loaded")
print("Classes:", dataset.classes)

# ----------------------------
# 6️⃣ LOAD PRETRAINED MODEL
# ----------------------------

model = models.resnet18(pretrained=True)

# Modify final layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

model = model.to(DEVICE)

# ----------------------------
# 7️⃣ LOSS & OPTIMIZER
# ----------------------------

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ----------------------------
# 8️⃣ TRAINING LOOP
# ----------------------------

print("🚀 Training started...")

for epoch in range(EPOCHS):
    running_loss = 0.0

    for images, labels in dataloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss:.4f}")

print("✅ Training completed")

# ----------------------------
# 9️⃣ SAVE MODEL
# ----------------------------

torch.save(model.state_dict(), MODEL_SAVE_PATH)
print("💾 Model saved as", MODEL_SAVE_PATH)
