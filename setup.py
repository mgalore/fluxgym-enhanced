#!/usr/bin/env python3
"""
Setup script for FluxGym LoRA Training Interface
This script helps users configure the application securely.
"""

import os
import json
import shutil
from pathlib import Path

def create_config_file():
    """Create a config.json file with user input."""
    print("🔧 FluxGym Setup - Configuration")
    print("=" * 50)
    
    config_path = "config.json"
    
    if os.path.exists(config_path):
        response = input(f"⚠️  {config_path} already exists. Overwrite? (y/N): ").strip().lower()
        if response != 'y':
            print("Setup cancelled.")
            return False
    
    print("\n📋 Please provide the following information:")
    
    # Get Google API key
    print("\n🔑 Google API Key:")
    print("   Get your key from: https://makersuite.google.com/app/apikey")
    api_key = input("   Enter your Google API key: ").strip()
    
    if not api_key:
        print("❌ API key is required. Setup cancelled.")
        return False
    
    # Optional settings
    print("\n⚙️  Optional Settings (press Enter for defaults):")
    
    max_images = input("   Max images per training (default: 150): ").strip()
    max_images = int(max_images) if max_images.isdigit() else 150
    
    resolution = input("   Default resolution (default: 512): ").strip()
    resolution = int(resolution) if resolution.isdigit() else 512
    
    learning_rate = input("   Default learning rate (default: 8e-4): ").strip()
    learning_rate = learning_rate if learning_rate else "8e-4"
    
    # Create configuration
    config = {
        "api_keys": {
            "google_api_key": api_key
        },
        "settings": {
            "max_images": max_images,
            "default_resolution": resolution,
            "default_epochs": 16,
            "default_learning_rate": learning_rate,
            "default_network_dim": 4,
            "default_vram": "20G"
        },
        "paths": {
            "models_dir": "models",
            "outputs_dir": "outputs",
            "datasets_dir": "datasets",
            "captions_dir": "generated_captions"
        }
    }
    
    # Save configuration
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        print(f"\n✅ Configuration saved to {config_path}")
        return True
    except Exception as e:
        print(f"❌ Error saving configuration: {e}")
        return False

def setup_gitignore():
    """Setup .gitignore file for security."""
    gitignore_path = ".gitignore"
    template_path = "gitignore_template.txt"
    
    if not os.path.exists(template_path):
        print(f"⚠️  {template_path} not found. Skipping .gitignore setup.")
        return
    
    if os.path.exists(gitignore_path):
        response = input(f"⚠️  {gitignore_path} already exists. Overwrite? (y/N): ").strip().lower()
        if response != 'y':
            print("Skipping .gitignore setup.")
            return
    
    try:
        shutil.copy(template_path, gitignore_path)
        print(f"✅ Created {gitignore_path} for security")
    except Exception as e:
        print(f"❌ Error creating .gitignore: {e}")

def create_directories():
    """Create necessary directories."""
    directories = [
        "models",
        "outputs", 
        "datasets",
        "generated_captions"
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"📁 Created directory: {directory}")
        except Exception as e:
            print(f"❌ Error creating directory {directory}: {e}")

def validate_dependencies():
    """Check if required dependencies are installed."""
    print("\n🔍 Checking dependencies...")
    
    required_packages = [
        "gradio",
        "torch", 
        "PIL",
        "google.generativeai",
        "slugify",
        "yaml"
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_").split(".")[0])
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies are installed!")
    return True

def main():
    """Main setup function."""
    print("🚀 FluxGym LoRA Training Interface Setup")
    print("=" * 50)
    print("This script will help you configure the application securely.\n")
    
    # Step 1: Create config file
    if not create_config_file():
        return
    
    # Step 2: Setup security
    print("\n🔒 Setting up security files...")
    setup_gitignore()
    
    # Step 3: Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Step 4: Validate dependencies
    validate_dependencies()
    
    # Final instructions
    print("\n" + "=" * 50)
    print("🎉 Setup Complete!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Run: python app.py")
    print("2. Open: http://localhost:7860")
    print("3. Upload your images and start training!")
    print("\n💡 Tips:")
    print("- Keep your config.json file secure and private")
    print("- Read the README.md for detailed usage instructions")
    print("- Check the console for any error messages")
    print("\n🔗 Get Google API key: https://makersuite.google.com/app/apikey")

if __name__ == "__main__":
    main() 