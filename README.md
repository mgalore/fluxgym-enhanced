# FluxGym LoRA Training Interface

A secure and user-friendly Gradio interface for training LoRA (Low-Rank Adaptation) models with Flux, featuring AI-powered captioning using Google's Gemini API.

## Features

- 🔐 **Secure API Key Management** - Configuration file and environment variable support
- 🤖 **AI-Powered Captioning** - Structured caption generation using Google Gemini
- 🎨 **Streetwear-Focused** - Specialized prompts for fashion/streetwear datasets
- 📊 **Real-time Training Logs** - Monitor training progress with live logs
- 🔧 **Flexible Configuration** - VRAM-optimized settings (12G, 16G, 20G+)
- 📁 **Organized Outputs** - Automatic file organization and backup

## Security Features

- ✅ API keys stored separately from code
- ✅ Environment variable fallback support
- ✅ Configuration validation
- ✅ Error handling and logging
- ✅ No hardcoded credentials

## Quick Setup (Automated Installation)

### 🚀 Option 1: One-Click Installation (Recommended)

**For Windows:**
```bash
# Double-click run_install.bat OR run in command prompt:
run_install.bat
```

**For Linux/Mac:**
```bash
chmod +x run_install.sh
./run_install.sh
```

**For Google Colab:**
```python
# Run this in a Colab cell:
!python colab_install.py
```

### 🛠️ Option 2: Manual Installation Script
```bash
python install.py
```

This automated installer will:
- ✅ Clone sd-scripts repository
- ✅ Install all dependencies
- ✅ Download required model files (~15GB)
- ✅ Set up directory structure
- ✅ Configure the application
- ✅ Verify installation

## Manual Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

**Option A: Using config.json (Recommended)**
1. Copy the sample configuration:
   ```bash
   cp config.json config.sample.json  # Keep as backup
   ```
2. Edit `config.json` and add your Google API key:
   ```json
   {
     "api_keys": {
       "google_api_key": "YOUR_ACTUAL_API_KEY_HERE"
     }
   }
   ```

**Option B: Using Environment Variables**
1. Copy the environment example:
   ```bash
   cp env.example .env
   ```
2. Edit `.env` and add your API key:
   ```
   GOOGLE_API_KEY=your_actual_api_key_here
   ```

### 3. Get Google API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the key to your configuration

### 4. Run the Application

```bash
python app.py
```

The interface will be available at `http://localhost:7860`

## Installation Scripts

| Script | Purpose | Environment |
|--------|---------|-------------|
| `install.py` | Complete automated setup | Local/Colab |
| `colab_install.py` | Colab-optimized installer | Google Colab |
| `run_install.bat` | Windows launcher | Windows |
| `run_install.sh` | Unix launcher | Linux/Mac |
| `setup.py` | Interactive configuration | All |
| `quick_start.py` | Pre-flight checks + launch | All |

## Usage Guide

### Step 1: Configure LoRA Settings
- **LoRA Name**: Choose a unique name for your model
- **Trigger Word**: Define the activation word/phrase
- **VRAM Setting**: Select based on your GPU (12G/16G/20G+)
- **Training Parameters**: Adjust epochs, learning rate, etc.

### Step 2: Upload Dataset
- Upload 4-30 images (ideal range)
- Optionally include .txt caption files
- Use the AI captioning feature for automatic structured captions

### Step 3: AI Captioning (Optional)
- Click "Add AI captions with Gemini"
- The system generates structured 16-part captions optimized for fashion/streetwear
- Captions follow the format: subject, framing, pose, outfit, style, colors, etc.

### Step 4: Start Training
- Review the generated training script and config
- Click "Start training"
- Monitor progress in real-time logs

## Configuration Options

### VRAM Optimization
- **20G+**: Full AdamW8bit optimizer
- **16G**: Adafactor with memory optimizations
- **12G**: Split mode training for lower VRAM

### File Structure
```
project/
├── app.py                 # Main application
├── config_loader.py       # Configuration management
├── config.json           # API keys and settings
├── requirements.txt      # Python dependencies
├── install.py            # Automated installer
├── colab_install.py      # Colab installer
├── setup.py              # Interactive setup
├── quick_start.py        # Launch with checks
├── models/               # Model files (auto-downloaded)
├── outputs/              # Training outputs
├── datasets/             # Processed datasets
└── generated_captions/   # AI-generated captions
```

## Model Downloads

The installer automatically downloads these models (~15GB total):

| Model | Size | Purpose |
|-------|------|---------|
| flux1-dev-fp8.safetensors | ~8GB | Main FLUX model |
| clip_l.safetensors | ~1GB | CLIP text encoder |
| t5xxl_fp8.safetensors | ~5GB | T5 text encoder |
| ae.sft | ~1GB | VAE autoencoder |

## Security Best Practices

1. **Never commit config.json** - Add to .gitignore
2. **Use environment variables** in production
3. **Rotate API keys** regularly
4. **Backup configurations** before changes
5. **Monitor API usage** for unexpected charges

## Troubleshooting

### Configuration Errors
```bash
Error: API key for 'google_api_key' not found
```
**Solution**: Check your config.json or set GOOGLE_API_KEY environment variable

### Import Errors
```bash
ImportError: config_loader.py not found
```
**Solution**: Ensure config_loader.py is in the same directory as app.py

### Installation Issues
```bash
# Re-run the installer
python install.py

# Or check installation
python quick_start.py
```

### VRAM Issues
- Reduce batch size or use lower VRAM setting
- Enable gradient checkpointing
- Use split mode for 12G VRAM

## File Security

### .gitignore Protection
Copy `gitignore_template.txt` to `.gitignore` to protect sensitive files:
```bash
cp gitignore_template.txt .gitignore
```

### Environment Variables Supported
- `GOOGLE_API_KEY`
- `GOOGLE_GENERATIVE_AI_API_KEY` 
- `GEMINI_API_KEY`
- `LOG_LEVEL`

## Advanced Configuration

Edit `config.json` to customize:
- File paths
- Default training parameters
- Model directories
- Output locations

Example:
```json
{
  "api_keys": {
    "google_api_key": "your-key-here"
  },
  "settings": {
    "max_images": 150,
    "default_resolution": 512,
    "default_learning_rate": "8e-4"
  },
  "paths": {
    "models_dir": "custom_models",
    "outputs_dir": "custom_outputs"
  }
}
```

## Support

For issues:
1. Check the console logs for detailed error messages
2. Verify API key configuration
3. Ensure all dependencies are installed
4. Check VRAM requirements for your GPU
5. Try re-running the installer: `python install.py`

## License

[Add your license information here]

<h1 align="center">Flux Gym Colab</h1>

![screenshot.png](screenshot.png) <br /> <br /> <br />


<h1>🐣 Please follow me for new updates</h1> 
Discord - (https://discord.gg/ES9nXE8z) <br />
X - (https://x.com/TheLocalLab_) <br />
Youtube - (https://www.youtube.com/@TheLocalLab) <br />
Patreon - (https://www.patreon.com/TheLocalLab)<br /> <br /> <br />


| Notebook | Info |
| --- | --- |
| <a href="https://colab.research.google.com/drive/1bG2RmkOVLVFPGsEm1RQIn5zsk2t3NRWS?usp=sharing" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | Flux Gym Colab |
| <a href="https://get.runpod.io/FluxGym-Template" target="_blank"><img src="https://6aa9b44fc527d43905e1f8d16e7ec46e1382d406-m.proxy3.startpage.com/pj/epqovs/tdpe/ST/matxGNBf64lNRFusGow32rMtsQ//////////pods/storage/SURFLYROOT//////////img/logo.svg?SURFLY=R" alt="Open Runpod Template" width="100"/></a> | Flux Gym Runpod Template |


<br /> <br />For those who prefer visual guidance, I've created a comprehensive step-by-step video tutorial demonstrating training Flux Lora Models with this Flux Gym Colab. This guide will walk you through the settings and steps to creating your own Loras. <br /> <br />

|                                           ***Tutorial Link***                                              |   
| :------------------------------------------------------------------------------------------------------: | 
| [![Watch the video](https://img.youtube.com/vi/yvXOKHeZtgs/hqdefault.jpg)](https://youtu.be/yvXOKHeZtgs) |


If you encounter any issues or have questions specific to the colab, feel free to reach out on [discord](https://discord.gg/5hmB4N4JFc), and I'll do my best to assist you.

## Credit
[Cocktailpeanut GitHub](https://github.com/cocktailpeanut/fluxgym)

[Ostris GitHub](https://github.com/ostris/ai-toolkit)

