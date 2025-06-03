# FluxGym Enhanced

Enhanced FluxGym with pretrained adapter support, bulk captioning, and optimized large dataset handling.

## 🚀 New Features

### ✨ Pretrained Adapter Support
- Load existing LoRA adapters for continued training
- Continue training from checkpoints
- Fine-tune existing models with new concepts

### 🤖 Bulk AI Captioning
- Generate captions for 300+ images using Google Gemini
- Automatic caption creation with trigger words
- Structured caption format for optimal training

### 📁 Large Dataset Optimization
- Handle 300+ images efficiently
- Direct dataset folder support (bypass slow uploads)
- Optimized training settings for large datasets

### 🛠️ Enhanced Tools
- `create_captions.py` - Bulk caption creation script
- Auto-detection of existing images in dataset folders
- Improved UI with helpful guidance

## 📋 Quick Start for Large Datasets

### 1. Setup Dataset Folder
```bash
# Copy your images directly to:
datasets/your-lora-name/
```

### 2. Auto-Caption with AI
- Open FluxGym web interface
- Enter LoRA name (matching folder name)
- Leave upload empty
- Click "Add AI captions with Gemini"

### 3. Optimized Training Settings for 300+ Images
- **Repeats**: 1-2 (instead of 10)
- **Epochs**: 8-12 (instead of 16)
- **Learning Rate**: 4e-4 or 2e-4 (lower)
- **LoRA Rank**: 8-16 (higher)

## 🔧 Installation

```bash
git clone https://github.com/mgalore/fluxgym-enhanced.git
cd fluxgym-enhanced
pip install -r requirements.txt
python app.py
```

## 📚 Enhanced Features

### Pretrained Adapter Training
```python
# Modified gen_sh function now supports:
--network_weights /path/to/existing/adapter.safetensors
```

### Bulk Caption Creation
```bash
# Create captions for all images in dataset folder
python create_captions.py datasets/my-lora my_trigger_word
```

### Large Dataset Workflow
1. Copy 300+ images to `datasets/lora-name/`
2. Use AI captioning (no upload needed)
3. Train with optimized settings
4. Monitor progress with frequent checkpoints

## ⚙️ Configuration

Enhanced `config.json` with:
- Configurable max image limits
- Path customization
- API key management

## 🎯 Best Practices

### For 300+ Images:
- Use **2 repeats** maximum
- Lower learning rate (**4e-4**)
- Higher LoRA rank (**12-16**)
- Save checkpoints every **2-4 epochs**

### Caption Quality:
- Use descriptive trigger words
- Include style, pose, lighting details
- Consistent caption structure

## 🐛 Troubleshooting

### Slow Uploads
- Use direct dataset folder method
- Copy images via file explorer
- Skip web interface uploads

### Memory Issues
- Reduce batch size
- Use 12G VRAM settings
- Enable gradient checkpointing

## 📖 Documentation

- [Original FluxGym](https://github.com/cocktailpeanut/fluxgym)
- [LoRA Training Guide](docs/lora-training.md)
- [Large Dataset Tips](docs/large-datasets.md)

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

Same as original FluxGym project.

---

**Enhanced by AI Assistant with focus on production-ready large dataset training.**