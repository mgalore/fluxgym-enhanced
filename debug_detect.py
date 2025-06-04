#!/usr/bin/env python3
"""
Debug script to test image detection functionality
"""

import os
from slugify import slugify

def debug_detect_images(lora_name="DailyPaperStyle"):
    """Debug function to test image detection."""
    
    print(f"🔍 Debug: Detecting images for LoRA: {lora_name}")
    
    # Get dataset folder path
    datasets_dir = "datasets"  # Using relative path
    slugified_name = slugify(lora_name)
    dataset_folder = os.path.join(datasets_dir, slugified_name)
    
    print(f"📁 Dataset folder path: {dataset_folder}")
    print(f"📁 Absolute path: {os.path.abspath(dataset_folder)}")
    
    # Check if dataset folder exists
    if not os.path.exists(dataset_folder):
        print(f"❌ Dataset folder does not exist: {dataset_folder}")
        return False
    else:
        print(f"✅ Dataset folder exists: {dataset_folder}")
    
    # List all files in the folder
    try:
        all_files = os.listdir(dataset_folder)
        print(f"📋 All files in folder ({len(all_files)} total):")
        for file in all_files[:10]:  # Show first 10 files
            print(f"   - {file}")
        if len(all_files) > 10:
            print(f"   ... and {len(all_files) - 10} more files")
    except Exception as e:
        print(f"❌ Error listing files: {e}")
        return False
    
    # Find image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    image_files = []
    
    for file in all_files:
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
    
    print(f"🖼️ Found {len(image_files)} image files:")
    for img in image_files[:5]:  # Show first 5 images
        print(f"   - {img}")
    if len(image_files) > 5:
        print(f"   ... and {len(image_files) - 5} more images")
    
    return len(image_files) > 0

if __name__ == "__main__":
    debug_detect_images() 