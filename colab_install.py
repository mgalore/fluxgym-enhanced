#!/usr/bin/env python3
"""
Google Colab Installation Script for FluxGym
Run this in a Colab cell to set up everything automatically
"""

import os
import subprocess
import urllib.request
import sys
import json

def run_cmd(command):
    """Run command and print output."""
    print(f"🔄 Running: {command}")
    result = os.system(command)
    if result == 0:
        print("✅ Success!")
    else:
        print("❌ Failed!")
    return result == 0

def download_model(url, path, name):
    """Download model with progress."""
    print(f"📥 Downloading {name}...")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    def progress(block_num, block_size, total_size):
        if total_size > 0:
            percent = min(100, (block_num * block_size * 100) // total_size)
            sys.stdout.write(f"\r{name}: {percent}% ")
            sys.stdout.flush()
    
    try:
        urllib.request.urlretrieve(url, path, progress)
        print(f"\n✅ {name} downloaded!")
        return True
    except Exception as e:
        print(f"\n❌ Failed to download {name}: {e}")
        return False

def main():
    """Main installation for Colab."""
    print("🚀 FluxGym Colab Installation Starting...")
    print("=" * 50)
    
    # Step 1: Navigate to correct directory
    print("\n📁 Setting up workspace...")
    os.chdir("/content")
    
    # Clone the repository if it doesn't exist
    if not os.path.exists("fluxgym-Colab"):
        print("📦 Repository not found. Please ensure fluxgym-Colab is cloned first.")
        print("Run: !git clone YOUR_REPO_URL fluxgym-Colab")
        return
    
    os.chdir("/content/fluxgym-Colab")
    print(f"✅ Working in: {os.getcwd()}")
    
    # Step 2: Clone sd-scripts
    print("\n📦 Cloning sd-scripts...")
    if os.path.exists("sd-scripts"):
        run_cmd("rm -rf sd-scripts")
    run_cmd("git clone -b sd3 https://github.com/kohya-ss/sd-scripts")
    
    # Step 3: Install sd-scripts requirements
    print("\n📦 Installing sd-scripts requirements...")
    os.chdir("/content/fluxgym-Colab/sd-scripts")
    run_cmd("pip install -r requirements.txt")
    
    # Step 4: Install main requirements
    print("\n📦 Installing main requirements...")
    os.chdir("/content/fluxgym-Colab")
    run_cmd("pip install -r requirements.txt")
    
    # Step 5: Install PyTorch
    print("\n🔥 Installing PyTorch 2.4...")
    run_cmd("pip install --pre torch==2.4 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    
    # Step 6: Create directories
    print("\n📁 Creating model directories...")
    os.makedirs("models/unet", exist_ok=True)
    os.makedirs("models/clip", exist_ok=True) 
    os.makedirs("models/vae", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("datasets", exist_ok=True)
    os.makedirs("generated_captions", exist_ok=True)
    
    # Step 7: Download models
    print("\n📥 Downloading model files...")
    models = [
        {
            "url": "https://huggingface.co/Kijai/flux-fp8/resolve/main/flux1-dev-fp8.safetensors",
            "path": "/content/fluxgym-Colab/models/unet/flux1-dev-fp8.safetensors",
            "name": "Flux UNET"
        },
        {
            "url": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors",
            "path": "/content/fluxgym-Colab/models/clip/clip_l.safetensors",
            "name": "CLIP L"
        },
        {
            "url": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors",
            "path": "/content/fluxgym-Colab/models/clip/t5xxl_fp8.safetensors", 
            "name": "T5XXL FP8"
        },
        {
            "url": "https://huggingface.co/cocktailpeanut/xulf-dev/resolve/main/ae.sft",
            "path": "/content/fluxgym-Colab/models/vae/ae.sft",
            "name": "VAE"
        }
    ]
    
    for model in models:
        if not os.path.exists(model["path"]):
            download_model(model["url"], model["path"], model["name"])
        else:
            print(f"✅ {model['name']} already exists")
    
    # Step 8: Fix Gradio version
    print("\n🔧 Fixing Gradio version...")
    run_cmd("pip uninstall gradio -y")
    run_cmd("pip install gradio==5.23.2")
    
    # Step 9: Setup config if needed
    config_path = "/content/fluxgym-Colab/config.json"
    if not os.path.exists(config_path):
        print("\n⚙️ Creating configuration file...")
        config = {
            "api_keys": {
                "google_api_key": "YOUR_GOOGLE_API_KEY_HERE"
            },
            "settings": {
                "max_images": 150,
                "default_resolution": 512,
                "default_epochs": 16,
                "default_learning_rate": "8e-4",
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
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        print("✅ Configuration file created")
    else:
        print("✅ Configuration already exists")
    
    # Final steps
    print("\n" + "=" * 50)
    print("🎉 Installation Complete!")
    print("=" * 50)
    print("\n📋 Next Steps:")
    print("1. 🔑 Set your Google API key in config.json")
    print("2. 🚀 Run: %cd /content/fluxgym-Colab")
    print("3. 🚀 Run: !python app.py")
    print("4. 🌐 Click the Gradio public URL")
    print("\n💡 Get API key: https://makersuite.google.com/app/apikey")
    print("✨ You're ready to train LoRA models!")

if __name__ == "__main__":
    main() 