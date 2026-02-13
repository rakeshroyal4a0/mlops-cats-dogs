import os
from PIL import Image

data_dir = "data/raw/archive/PetImages"

for category in ["Cat", "Dog"]:
    folder_path = os.path.join(data_dir, category)

    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        continue

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        try:
            img = Image.open(file_path)
            img.verify()
        except Exception:
            print(f"Removing corrupted file: {file_path}")
            os.remove(file_path)

print("Cleaning completed.")
