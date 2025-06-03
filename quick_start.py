#!/usr/bin/env python3
"""
Quick Start script for FluxGym LoRA Training Interface
Checks configuration and launches the application.
"""

import os
import sys
import subprocess

def check_config():
    """Check if configuration is properly set up."""
    print("🔍 Checking configuration...")
    
    # Check if config_loader exists
    if not os.path.exists("config_loader.py"):
        print("❌ config_loader.py not found!")
        return False
    
    # Check if config.json exists and is valid
    if not os.path.exists("config.json"):
        print("❌ config.json not found!")
        print("💡 Run: python setup.py")
        return False
    
    # Try to load configuration
    try:
        from config_loader import config
        # Test API key retrieval
        api_key = config.get_api_key("google_api_key")
        if api_key == "YOUR_GOOGLE_API_KEY_HERE":
            print("❌ Google API key not configured!")
            print("💡 Edit config.json or set GOOGLE_API_KEY environment variable")
            return False
        print("✅ Configuration looks good!")
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    required = ["gradio", "torch", "PIL", "google.generativeai"]
    missing = []
    
    for package in required:
        try:
            __import__(package.replace("-", "_").split(".")[0])
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("💡 Run: pip install -r requirements.txt")
        return False
    
    print("✅ Dependencies are installed!")
    return True

def launch_app():
    """Launch the Gradio application."""
    print("🚀 Launching FluxGym...")
    
    try:
        # Import and launch the app
        import app
        print("✅ Application launched successfully!")
        print("🌐 Open: http://localhost:7860")
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error launching application: {e}")
        print("💡 Check the console for detailed error messages")

def main():
    """Main function."""
    print("⚡ FluxGym Quick Start")
    print("=" * 30)
    
    # Step 1: Check configuration
    if not check_config():
        print("\n🔧 Configuration issues found.")
        print("Run 'python setup.py' to configure the application.")
        return
    
    # Step 2: Check dependencies
    if not check_dependencies():
        print("\n📦 Dependency issues found.")
        print("Run 'pip install -r requirements.txt' to install dependencies.")
        return
    
    # Step 3: Launch application
    print("\n🎯 All checks passed!")
    launch_app()

if __name__ == "__main__":
    main() 