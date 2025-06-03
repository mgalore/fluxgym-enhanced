#!/usr/bin/env python3
"""
Bulk Caption Creator for FluxGym
Creates basic caption files for images in dataset folder
"""

import os
import sys
from pathlib import Path

def create_captions(dataset_folder, trigger_word, base_caption=None):
    """Create caption files for all images in dataset folder."""
    
    if not os.path.exists(dataset_folder):
        print(f"Error: Dataset folder '{dataset_folder}' does not exist")
        return
    
    # Default caption template
    if not base_caption:
        base_caption = f"person wearing {trigger_word} style, full body, standing pose, casual clothing, outdoor setting, natural lighting, high quality photo"
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    
    created_count = 0
    skipped_count = 0
    
    # Process all images in folder
    for file_path in Path(dataset_folder).iterdir():
        if file_path.suffix.lower() in image_extensions:
            # Check if caption file already exists
            caption_path = file_path.with_suffix('.txt')
            
            if caption_path.exists():
                print(f"Skipped: {file_path.name} (caption already exists)")
                skipped_count += 1
                continue
            
            # Create caption file
            with open(caption_path, 'w', encoding='utf-8') as f:
                f.write(base_caption)
            
            print(f"Created: {caption_path.name}")
            created_count += 1
    
    print(f"\nSummary:")
    print(f"Created: {created_count} caption files")
    print(f"Skipped: {skipped_count} files (already had captions)")
    print(f"Total images: {created_count + skipped_count}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python create_captions.py <dataset_folder> <trigger_word> [custom_caption]")
        print("\nExample:")
        print("  python create_captions.py datasets/my-lora streetwear_core")
        print("  python create_captions.py datasets/my-lora vintage_style 'person in vintage_style clothing, retro fashion, classic pose'")
        return
    
    dataset_folder = sys.argv[1]
    trigger_word = sys.argv[2]
    custom_caption = sys.argv[3] if len(sys.argv) > 3 else None
    
    create_captions(dataset_folder, trigger_word, custom_caption)

if __name__ == "__main__":
    main() 