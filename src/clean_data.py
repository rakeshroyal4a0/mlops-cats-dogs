import os
from PIL import Image

# Find correct PetImages folder automatically
base_path = "data/raw"

# search for PetImages folder dynamically
for root, dirs, files in os.walk(base_path):
    if "Cat" in dirs and "Dog" in dirs:
        data_dir = root
        break

print(f"Using dataset path: {data_dir}")

for category in ["Cat", "Dog"]:
    folder_path = os.path.join(data_dir, category)

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        try:
            img = Image.open(file_path)
            img.verify()
        except Exception:
            print(f"Removing corrupted file: {file_path}")
            os.remove(file_path)

print("Cleaning completed.")
