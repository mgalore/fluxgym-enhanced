import os
from slugify import slugify

# Test the exact folder structure
lora_name = "streetwear_core"
slugified = slugify(lora_name)
dataset_folder = os.path.join("datasets", slugified)

print(f"LoRA name: {lora_name}")
print(f"Slugified: {slugified}")
print(f"Dataset folder: {dataset_folder}")
print(f"Exists: {os.path.exists(dataset_folder)}")

if os.path.exists(dataset_folder):
    files = os.listdir(dataset_folder)
    print(f"Files: {files}")
    image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp'))]
    print(f"Image files: {image_files}")
else:
    print("Folder not found!") 