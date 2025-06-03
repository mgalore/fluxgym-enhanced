#!/usr/bin/env python3
"""
FluxGym Complete Installation Script
Automates the entire setup process for FluxGym LoRA Training Interface
"""

import os
import sys
import subprocess
import urllib.request
import json
import shutil
from pathlib import Path
import time

class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_step(step_num, total_steps, description):
    """Print a formatted step indicator."""
    print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}Step {step_num}/{total_steps}: {description}{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")

def print_success(message):
    """Print a success message."""
    print(f"{Colors.OKGREEN}✅ {message}{Colors.ENDC}")

def print_error(message):
    """Print an error message."""
    print(f"{Colors.FAIL}❌ {message}{Colors.ENDC}")

def print_warning(message):
    """Print a warning message."""
    print(f"{Colors.WARNING}⚠️  {message}{Colors.ENDC}")

def print_info(message):
    """Print an info message."""
    print(f"{Colors.OKCYAN}ℹ️  {message}{Colors.ENDC}")

def is_colab():
    """Check if running in Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def run_command(command, description="", check=True):
    """Run a shell command with error handling."""
    if description:
        print_info(f"Running: {description}")
    
    print(f"{Colors.OKBLUE}$ {command}{Colors.ENDC}")
    
    try:
        result = subprocess.run(command, shell=True, check=check, 
                              capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed: {e}")
        if e.stderr:
            print(e.stderr)
        return False

def download_file(url, filepath, description=""):
    """Download a file with progress indication."""
    if description:
        print_info(f"Downloading: {description}")
    
    print(f"URL: {url}")
    print(f"Destination: {filepath}")
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    try:
        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) // total_size)
                sys.stdout.write(f"\rProgress: {percent}% ")
                sys.stdout.flush()
        
        urllib.request.urlretrieve(url, filepath, progress_hook)
        print()  # New line after progress
        print_success(f"Downloaded: {os.path.basename(filepath)}")
        return True
    except Exception as e:
        print_error(f"Download failed: {e}")
        return False

def setup_directories():
    """Create necessary directories."""
    directories = [
        "models/unet",
        "models/clip", 
        "models/vae",
        "outputs",
        "datasets",
        "generated_captions"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print_success(f"Created directory: {directory}")

def clone_sd_scripts():
    """Clone the sd-scripts repository."""
    if os.path.exists("sd-scripts"):
        print_warning("sd-scripts directory already exists, removing...")
        shutil.rmtree("sd-scripts")
    
    success = run_command(
        "git clone -b sd3 https://github.com/kohya-ss/sd-scripts",
        "Cloning sd-scripts repository"
    )
    
    if success:
        print_success("sd-scripts cloned successfully")
    return success

def install_sd_scripts_requirements():
    """Install sd-scripts requirements."""
    if not os.path.exists("sd-scripts"):
        print_error("sd-scripts directory not found")
        return False
    
    os.chdir("sd-scripts")
    success = run_command(
        "pip install -r requirements.txt",
        "Installing sd-scripts requirements"
    )
    os.chdir("..")
    
    if success:
        print_success("sd-scripts requirements installed")
    return success

def install_main_requirements():
    """Install main application requirements."""
    success = run_command(
        "pip install -r requirements.txt",
        "Installing main application requirements"
    )
    
    if success:
        print_success("Main requirements installed")
    return success

def install_pytorch():
    """Install specific PyTorch version."""
    success = run_command(
        "pip install --pre torch==2.4 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
        "Installing PyTorch 2.4 with CUDA 12.1 support"
    )
    
    if success:
        print_success("PyTorch installed")
    return success

def download_models():
    """Download required model files."""
    models = [
        {
            "url": "https://huggingface.co/Kijai/flux-fp8/resolve/main/flux1-dev-fp8.safetensors",
            "path": "models/unet/flux1-dev-fp8.safetensors",
            "description": "Flux UNET model"
        },
        {
            "url": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors",
            "path": "models/clip/clip_l.safetensors", 
            "description": "CLIP L model"
        },
        {
            "url": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors",
            "path": "models/clip/t5xxl_fp8.safetensors",
            "description": "T5XXL FP8 model"
        },
        {
            "url": "https://huggingface.co/cocktailpeanut/xulf-dev/resolve/main/ae.sft",
            "path": "models/vae/ae.sft",
            "description": "VAE model"
        }
    ]
    
    success = True
    for model in models:
        if os.path.exists(model["path"]):
            print_warning(f"Model already exists: {model['path']}")
            continue
            
        if not download_file(model["url"], model["path"], model["description"]):
            success = False
    
    return success

def fix_gradio_version():
    """Install specific Gradio version."""
    print_info("Uninstalling current Gradio version...")
    run_command("pip uninstall gradio -y", check=False)
    
    success = run_command(
        "pip install gradio==5.23.2",
        "Installing Gradio 5.23.2"
    )
    
    if success:
        print_success("Gradio version fixed")
    return success

def setup_configuration():
    """Set up configuration if it doesn't exist."""
    config_path = "config.json"
    
    if os.path.exists(config_path):
        print_warning("config.json already exists, skipping configuration setup")
        return True
    
    print_info("Setting up basic configuration...")
    
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
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        print_success("Configuration file created")
        return True
    except Exception as e:
        print_error(f"Failed to create configuration: {e}")
        return False

def check_installation():
    """Verify the installation."""
    print_info("Verifying installation...")
    
    # Check critical files
    critical_files = [
        "app.py",
        "config_loader.py",
        "config.json",
        "requirements.txt",
        "sd-scripts/flux_train_network.py"
    ]
    
    missing_files = []
    for file_path in critical_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print_error(f"Missing critical files: {missing_files}")
        return False
    
    # Check models
    model_files = [
        "models/unet/flux1-dev-fp8.safetensors",
        "models/clip/clip_l.safetensors",
        "models/clip/t5xxl_fp8.safetensors", 
        "models/vae/ae.sft"
    ]
    
    missing_models = []
    for model_path in model_files:
        if not os.path.exists(model_path):
            missing_models.append(model_path)
    
    if missing_models:
        print_warning(f"Missing model files: {missing_models}")
        print_info("Some models may not have downloaded completely. You can run this script again.")
    
    print_success("Installation verification complete")
    return True

def main():
    """Main installation function."""
    print(f"{Colors.BOLD}{Colors.HEADER}")
    print("🚀 FluxGym Complete Installation Script")
    print("=" * 60)
    print("This script will set up everything you need for FluxGym LoRA training")
    print("=" * 60)
    print(f"{Colors.ENDC}")
    
    # Detect environment
    if is_colab():
        print_info("Google Colab environment detected")
        # Change to the correct directory for Colab
        os.chdir("/content/fluxgym-Colab")
    else:
        print_info("Local environment detected")
    
    print_info(f"Working directory: {os.getcwd()}")
    
    steps = [
        ("Setup directories", setup_directories),
        ("Clone sd-scripts repository", clone_sd_scripts),
        ("Install sd-scripts requirements", install_sd_scripts_requirements),
        ("Install main requirements", install_main_requirements),
        ("Install PyTorch", install_pytorch),
        ("Download model files", download_models),
        ("Fix Gradio version", fix_gradio_version),
        ("Setup configuration", setup_configuration),
        ("Verify installation", check_installation)
    ]
    
    total_steps = len(steps)
    failed_steps = []
    
    for i, (description, function) in enumerate(steps, 1):
        print_step(i, total_steps, description)
        
        try:
            if not function():
                failed_steps.append(description)
                print_error(f"Step failed: {description}")
        except Exception as e:
            failed_steps.append(description)
            print_error(f"Step failed with exception: {description} - {e}")
    
    # Summary
    print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}Installation Summary{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")
    
    if failed_steps:
        print_warning(f"Some steps failed: {failed_steps}")
        print_info("You may need to run the script again or fix issues manually")
    else:
        print_success("All steps completed successfully!")
    
    print("\n" + "="*60)
    print(f"{Colors.BOLD}Next Steps:{Colors.ENDC}")
    print("1. Configure your Google API key in config.json")
    print("2. Run: python app.py")
    print("3. Open: http://localhost:7860")
    print("4. Start training your LoRA models!")
    print("\n💡 Tips:")
    print("- Get Google API key: https://makersuite.google.com/app/apikey")
    print("- Read README.md for detailed usage instructions")
    print("- Use 'python quick_start.py' for pre-flight checks")
    print("="*60)

if __name__ == "__main__":
    main() 