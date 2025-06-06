import os
import sys
import subprocess
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set HF environment variable
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Add the current working directory to the Python path
sys.path.insert(0, os.getcwd())

# Import configuration loader
try:
    from config_loader import config, ConfigurationError
except ImportError:
    logger.error("config_loader.py not found. Please ensure it exists in the same directory.")
    sys.exit(1)

# Auto-install google-generativeai if not available
try:
    import google.generativeai as genai
except ImportError:
    logger.info("Installing google-generativeai...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "google-generativeai"])
    import google.generativeai as genai

import gradio as gr
from PIL import Image
import torch
import uuid
import shutil
import json
import yaml
from slugify import slugify
# Removed gradio_logsview due to compatibility issues
# Using simple textbox for logging instead
from huggingface_hub import hf_hub_download

# Load configuration settings
try:
    GOOGLE_API_KEY = config.get_api_key("google_api_key")
    MAX_IMAGES = config.get_setting("max_images", 150)
    PRETRAINED_ADAPTERS_DIR = config.get_path("pretrained_adapters_dir", "pretrained_adapters")
    logger.info("Configuration loaded successfully")
except ConfigurationError as e:
    logger.error(f"Configuration error: {e}")
    logger.error("Please check your config.json file or set environment variables.")
    # Show helpful message to user
    print("\n" + "="*60)
    print("CONFIGURATION ERROR")
    print("="*60)
    print(f"Error: {e}")
    print("\nTo fix this issue:")
    print("1. Copy config.json and update with your Google API key, OR")
    print("2. Set environment variable: GOOGLE_API_KEY=your_api_key")
    print("3. You can get a Google API key from: https://makersuite.google.com/app/apikey")
    print("="*60)
    sys.exit(1)

# Create pretrained adapters directory if it doesn't exist
os.makedirs(PRETRAINED_ADAPTERS_DIR, exist_ok=True)

def load_captioning(uploaded_files, concept_sentence):
    """Load and process uploaded files for captioning."""
    try:
        uploaded_images = [file for file in uploaded_files if not file.endswith('.txt')]
        txt_files = [file for file in uploaded_files if file.endswith('.txt')]
        txt_files_dict = {os.path.splitext(os.path.basename(txt_file))[0]: txt_file for txt_file in txt_files}
        updates = []
        
        # Check for existing images in dataset folder if no uploads
        if len(uploaded_images) == 0:
            # Look for existing images in dataset folders
            datasets_dir = config.get_path("datasets_dir", "datasets")
            existing_images = []
            
            if os.path.exists(datasets_dir):
                for folder in os.listdir(datasets_dir):
                    folder_path = os.path.join(datasets_dir, folder)
                    if os.path.isdir(folder_path):
                        for file in os.listdir(folder_path):
                            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
                                existing_images.append(os.path.join(folder_path, file))
            
            if existing_images:
                uploaded_images = existing_images[:MAX_IMAGES]  # Limit to MAX_IMAGES
                gr.Info(f"Found {len(existing_images)} existing images in dataset folders. Using first {len(uploaded_images)} for captioning.")
        
        if len(uploaded_images) <= 1:
            raise gr.Error(
                "Please upload at least 2 images to train your model (the ideal number with default settings is between 4-30), or copy images directly to datasets/[lora-name]/ folder"
            )
        elif len(uploaded_images) > MAX_IMAGES:
            raise gr.Error(f"For now, only {MAX_IMAGES} or less images are allowed for training")
        
        # Update for the captioning_area
        updates.append(gr.update(visible=True))
        
        # Update visibility and image for each captioning row and image
        for i in range(1, MAX_IMAGES + 1):
            # Determine if the current row and image should be visible
            visible = i <= len(uploaded_images)

            # Update visibility of the captioning row
            updates.append(gr.update(visible=visible))

            # Update for image component - display image if available, otherwise hide
            image_value = uploaded_images[i - 1] if visible else None
            updates.append(gr.update(value=image_value, visible=visible))

            corresponding_caption = False
            if(image_value):
                base_name = os.path.splitext(os.path.basename(image_value))[0]
                # Check for existing caption file
                caption_file_path = os.path.join(os.path.dirname(image_value), f"{base_name}.txt")
                if os.path.exists(caption_file_path):
                    with open(caption_file_path, 'r', encoding='utf-8') as file:
                        corresponding_caption = file.read()
                elif base_name in txt_files_dict:
                    with open(txt_files_dict[base_name], 'r', encoding='utf-8') as file:
                        corresponding_caption = file.read()

            # Update value of captioning area
            text_value = corresponding_caption if visible and corresponding_caption else concept_sentence if visible and concept_sentence else None
            updates.append(gr.update(value=text_value, visible=visible))

        # Update for the sample caption area
        updates.append(gr.update(visible=True))
        updates.append(gr.update(visible=True))

        return updates
    except Exception as e:
        logger.error(f"Error in load_captioning: {e}")
        raise gr.Error(f"Error loading captioning: {str(e)}")

def hide_captioning():
    """Hide captioning interface."""
    return gr.update(visible=False), gr.update(visible=False)

def resize_image(image_path, output_path, size):
    """Resize image to specified size while maintaining aspect ratio."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            if width < height:
                new_width = size
                new_height = int((size/width) * height)
            else:
                new_height = size
                new_width = int((size/height) * width)
            logger.info(f"Resizing {image_path}: {new_width}x{new_height}")
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            img_resized.save(output_path)
    except Exception as e:
        logger.error(f"Error resizing image {image_path}: {e}")
        raise

def create_dataset(destination_folder, size, *inputs):
    """Create dataset from uploaded images and captions."""
    try:
        logger.info("Creating dataset")
        images = inputs[0]
        
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        # Check if images already exist in the destination folder
        existing_images = []
        if os.path.exists(destination_folder):
            for file in os.listdir(destination_folder):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
                    existing_images.append(os.path.join(destination_folder, file))
        
        # If images already exist in folder and no new uploads, use existing images
        if existing_images and (not images or len(images) == 0):
            logger.info(f"Found {len(existing_images)} existing images in {destination_folder}")
            logger.info("Using existing images in dataset folder")
            
            # Resize existing images if needed
            for image_path in existing_images:
                resize_image(image_path, image_path, size)
                
            return destination_folder

        # Process uploaded images (original behavior)
        for index, image in enumerate(images):
            # Copy the images to the datasets folder
            new_image_path = shutil.copy(image, destination_folder)

            # Resize the images
            resize_image(new_image_path, new_image_path, size)

            # Copy the captions
            original_caption = inputs[index + 1]

            image_file_name = os.path.basename(new_image_path)
            caption_file_name = os.path.splitext(image_file_name)[0] + ".txt"
            caption_path = resolve_path_without_quotes(os.path.join(destination_folder, caption_file_name))
            logger.info(f"Image: {new_image_path}, Caption: {caption_path}")
            
            with open(caption_path, 'w', encoding='utf-8') as file:
                file.write(original_caption)

        logger.info(f"Dataset created at: {destination_folder}")
        return destination_folder
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        raise gr.Error(f"Error creating dataset: {str(e)}")

def run_captioning(images, concept_sentence, *captions):
    """Generate captions using Google Gemini API."""
    try:
        logger.info("Starting Gemini AI captioning...")
        
        # Configure Google Gemini API
        genai.configure(api_key=GOOGLE_API_KEY)
        
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Load system prompt from file
        system_prompt_file = "system_prompt.txt"
        try:
            with open(system_prompt_file, 'r', encoding='utf-8') as f:
                system_prompt = f.read().strip()
            logger.info(f"Loaded system prompt from {system_prompt_file}")
        except FileNotFoundError:
            logger.warning(f"System prompt file {system_prompt_file} not found, using fallback")
            # Fallback prompt if file is missing
            system_prompt = """You are Caption-AI, a specialized assistant that converts street-fashion images into prompt-style captions for LoRA fine-tuning.
Follow the exact rules below for every image you receive.
Return one single-line, comma-separated caption, all lowercase, no periods.

Please analyze the image and write a caption that includes structured information based on the fashion image provided.
Include subject type, framing, pose, outfit description, clothing style tag, accessories, location, lighting, and camera details.
Keep the caption concise but descriptive for training purposes."""

        captions = list(captions)
        
        # Handle different sources of images
        image_paths = []
        
        if images is not None and images != [] and len(images) > 0:
            # If images were uploaded via file component
            logger.info("Processing uploaded images...")
            if isinstance(images, list):
                image_paths = [img.name if hasattr(img, 'name') else img for img in images]
            else:
                image_paths = [images.name if hasattr(images, 'name') else images]
        else:
            # If images are already in dataset (from detect_existing_images)
            logger.info("No uploaded images found, collecting from dataset display...")
            
            # We need to collect image paths from the dataset folder
            # Try to get images from the default dataset structure
            datasets_dir = "datasets"
            
            # Look for any dataset folder with images
            if os.path.exists(datasets_dir):
                for folder_name in os.listdir(datasets_dir):
                    folder_path = os.path.join(datasets_dir, folder_name)
                    if os.path.isdir(folder_path):
                        # Get all image files in this folder
                        for file in os.listdir(folder_path):
                            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
                                image_paths.append(os.path.join(folder_path, file))
                        
                        # If we found images in this folder, use them
                        if image_paths:
                            logger.info(f"Found {len(image_paths)} images in dataset folder: {folder_path}")
                            break
            
            if not image_paths:
                raise gr.Error("No images found! Please upload images or use 'Detect Existing Images' first.")
        
        logger.info(f"Processing {len(image_paths)} images for captioning...")
        
        # Create captions directory if it doesn't exist
        captions_dir = config.get_path("captions_dir", "generated_captions")
        os.makedirs(captions_dir, exist_ok=True)
        
        # Prepare master captions file
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        master_captions_file = os.path.join(captions_dir, f"all_captions_{timestamp}.txt")
        
        # List to store all captions for the master file
        all_captions_data = []
        
        logger.info("🚀 Starting Gemini AI captioning...")
        
        for i, image_path in enumerate(image_paths):
            logger.info(f"Processing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            if isinstance(image_path, str):  # If image is a file path
                image_pil = Image.open(image_path).convert("RGB")

            try:
                # Prepare the prompt with concept sentence if provided
                final_prompt = system_prompt
                if concept_sentence:
                    final_prompt += f"\n\nNote: Use '{concept_sentence}' as the clothing style tag and trigger word."
                
                # Generate caption using Gemini
                logger.info(f"📸 Analyzing image with Gemini...")
                response = model.generate_content([final_prompt, image_pil])
                
                if response.text:
                    structured_caption = response.text.strip()
                    
                    # Clean up the response (remove any extra formatting)
                    structured_caption = structured_caption.replace('\n', ' ').replace('\r', ' ')
                    # Remove any markdown formatting
                    structured_caption = structured_caption.replace('**', '').replace('*', '')
                    
                    logger.info(f"✅ Caption generated: {structured_caption[:100]}...")
                    
                else:
                    # Fallback if Gemini doesn't respond
                    structured_caption = f"person, full-body, standing, casual wear, {concept_sentence if concept_sentence else 'streetwear_core'}, neutral tones, cotton, eye-level, DSLR, portrait lens, natural light, daylight, outdoor, urban setting, solo, lookbook style, casual vibe, {concept_sentence if concept_sentence else 'streetwear_core'}"
                    logger.warning("⚠️ Gemini didn't provide a response, using fallback caption")
                    
            except Exception as e:
                logger.error(f"❌ Error with Gemini API: {e}")
                # Fallback caption
                structured_caption = f"person, full-body, standing, casual wear, {concept_sentence if concept_sentence else 'streetwear_core'}, neutral tones, cotton, eye-level, DSLR, portrait lens, natural light, daylight, outdoor, urban setting, solo, lookbook style, casual vibe, {concept_sentence if concept_sentence else 'streetwear_core'}"
            
            # Update the caption in the UI if we have enough caption fields
            if i < len(captions):
                captions[i] = structured_caption
            
            # Save individual caption file in the same directory as the image
            image_filename = os.path.basename(image_path)
            image_name = os.path.splitext(image_filename)[0]
            image_dir = os.path.dirname(image_path)
            
            # Save .txt file alongside the image
            caption_filepath = os.path.join(image_dir, f"{image_name}.txt")
            
            with open(caption_filepath, 'w', encoding='utf-8') as f:
                f.write(structured_caption)
            
            # Also save in captions directory for backup
            caption_filename = f"{image_name}_caption.txt"
            backup_caption_filepath = os.path.join(captions_dir, caption_filename)
            
            with open(backup_caption_filepath, 'w', encoding='utf-8') as f:
                f.write(structured_caption)
            
            # Store data for master file
            all_captions_data.append({
                'image_path': image_path,
                'image_filename': image_filename,
                'structured_caption': structured_caption
            })
            
            logger.info(f"💾 Saved caption to: {caption_filepath}")

            yield captions
        
        # Save master captions file with all captions
        with open(master_captions_file, 'w', encoding='utf-8') as f:
            f.write("# Generated Captions Report (Gemini)\n")
            f.write(f"# Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# API Key used: {GOOGLE_API_KEY[:20]}...\n")
            f.write(f"# Concept sentence: {concept_sentence}\n")
            f.write(f"# Total images processed: {len(all_captions_data)}\n")
            f.write(f"# Format: Image Filename | Structured Caption\n")
            f.write("="*80 + "\n\n")
            
            for data in all_captions_data:
                f.write(f"Image: {data['image_filename']}\n")
                f.write(f"Path: {data['image_path']}\n")
                f.write(f"Structured Caption: {data['structured_caption']}\n")
                f.write("-" * 80 + "\n\n")
            
            # Also create a simple format for easy copy-paste
            f.write("\n" + "="*80 + "\n")
            f.write("# SIMPLE FORMAT (for easy copy-paste)\n")
            f.write("="*80 + "\n\n")
            
            for data in all_captions_data:
                f.write(f"{data['image_filename']}: {data['structured_caption']}\n")
        
        logger.info(f"🎉 Master captions file saved to: {master_captions_file}")
        logger.info(f"📁 Individual caption files saved in: {captions_dir}")
        logger.info(f"💾 Caption .txt files saved alongside images in dataset folder")
        logger.info("✨ Gemini captioning completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in run_captioning: {e}")
        logger.error(traceback.format_exc())
        raise gr.Error(f"Error running captioning: {str(e)}")

def convert_to_structured_format(raw_caption, concept_sentence):
    """
    Convert Florence's natural language description to the structured 16-part format
    This is a simplified conversion - you may want to enhance this with more sophisticated parsing
    """
    
    # Default values for the 16 fields
    fields = {
        'subject': 'person',
        'framing': 'full-body',
        'pose': 'standing',
        'outfit': 'casual wear',
        'style_tag': concept_sentence if concept_sentence else 'streetwear_core',
        'color': 'neutral tones',
        'texture': 'cotton',
        'angle': 'eye-level',
        'camera': 'DSLR',
        'lens': 'portrait lens',
        'lighting': 'natural light',
        'time': 'daylight',
        'location': 'outdoor',
        'backdrop': 'urban setting',
        'composition': 'solo',
        'style': 'lookbook style'
    }
    
    # Simple keyword matching to extract information from raw caption
    raw_lower = raw_caption.lower()
    
    # Subject detection
    if 'woman' in raw_lower or 'female' in raw_lower:
        fields['subject'] = 'female model'
    elif 'man' in raw_lower or 'male' in raw_lower:
        fields['subject'] = 'male model'
    elif 'people' in raw_lower or 'group' in raw_lower:
        fields['subject'] = 'group'
    
    # Framing detection
    if 'close-up' in raw_lower or 'portrait' in raw_lower:
        fields['framing'] = 'close-up'
    elif 'full body' in raw_lower or 'full-body' in raw_lower:
        fields['framing'] = 'full-body'
    elif 'back' in raw_lower:
        fields['framing'] = 'back view'
    
    # Pose detection
    if 'sitting' in raw_lower:
        fields['pose'] = 'sitting'
    elif 'walking' in raw_lower:
        fields['pose'] = 'walking'
    elif 'arms crossed' in raw_lower:
        fields['pose'] = 'arms crossed'
    elif 'hands in pockets' in raw_lower:
        fields['pose'] = 'hands in pockets'
    
    # Outfit detection
    if 'hoodie' in raw_lower:
        fields['outfit'] = 'hoodie'
    elif 't-shirt' in raw_lower or 'tee' in raw_lower:
        fields['outfit'] = 'tee'
    elif 'jacket' in raw_lower:
        fields['outfit'] = 'jacket'
    elif 'dress' in raw_lower:
        fields['outfit'] = 'dress'
    
    # Color detection
    if 'black' in raw_lower:
        fields['color'] = 'black palette'
    elif 'white' in raw_lower:
        fields['color'] = 'white palette'
    elif 'blue' in raw_lower:
        fields['color'] = 'blue palette'
    elif 'red' in raw_lower:
        fields['color'] = 'red palette'
    elif 'monochrome' in raw_lower:
        fields['color'] = 'monochrome'
    
    # Location detection
    if 'indoor' in raw_lower or 'inside' in raw_lower:
        fields['location'] = 'indoor'
    elif 'studio' in raw_lower:
        fields['location'] = 'studio'
    elif 'street' in raw_lower:
        fields['location'] = 'street'
    elif 'alley' in raw_lower:
        fields['location'] = 'alleyway'
    
    # Lighting detection
    if 'bright' in raw_lower or 'well-lit' in raw_lower:
        fields['lighting'] = 'studio light'
    elif 'dark' in raw_lower or 'dim' in raw_lower:
        fields['lighting'] = 'low light'
    elif 'flash' in raw_lower:
        fields['lighting'] = 'high flash'
    
    # Generate extras based on raw caption
    extras = []
    if 'confident' in raw_lower:
        extras.append('confident posture')
    if 'moody' in raw_lower or 'dramatic' in raw_lower:
        extras.append('moody atmosphere')
    if 'urban' in raw_lower:
        extras.append('urban vibe')
    
    # If no extras found, add some generic ones
    if not extras:
        extras = ['stylish look', 'contemporary feel']
    
    # Construct the structured caption
    structured_parts = [
        fields['subject'],
        fields['framing'],
        fields['pose'],
        fields['outfit'],
        fields['style_tag'],
        fields['color'],
        fields['texture'],
        fields['angle'],
        fields['camera'],
        fields['lens'],
        fields['lighting'],
        fields['time'],
        fields['location'],
        fields['backdrop'],
        fields['composition'],
        fields['style']
    ]
    
    # Add extras
    structured_parts.extend(extras[:3])  # Limit to 3 extras
    
    # Add trigger word
    trigger_word = concept_sentence if concept_sentence else 'streetwear_core'
    structured_parts.append(trigger_word)
    
    return ', '.join(structured_parts)

def recursive_update(d, u):
    """Recursively update dictionary."""
    for k, v in u.items():
        if isinstance(v, dict) and v:
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def resolve_path(p):
    """Resolve path with quotes for command line usage."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    norm_path = os.path.normpath(os.path.join(current_dir, p))
    return f"\"{norm_path}\""

def resolve_path_without_quotes(p):
    """Resolve path without quotes."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    norm_path = os.path.normpath(os.path.join(current_dir, p))
    return norm_path

def gen_sh(
    output_name,
    resolution,
    seed,
    workers,
    learning_rate,
    network_dim,
    max_train_epochs,
    save_every_n_epochs,
    timestep_sampling,
    guidance_scale,
    vram,
    sample_prompts,
    sample_every_n_steps,
    network_weights=None,
):
    """Generate shell script for training."""
    try:
        logger.info(f"Generating training script: network_dim={network_dim}, epochs={max_train_epochs}, vram={vram}")

        line_break = "\\"
        file_type = "sh"
        if sys.platform == "win32":
            line_break = "^"
            file_type = "bat"

        sample = ""
        if len(sample_prompts) > 0 and sample_every_n_steps > 0:
            sample = f"""--sample_prompts={resolve_path('sample_prompts.txt')} --sample_every_n_steps="{sample_every_n_steps}" {line_break}"""

        # Use configurable paths
        models_dir = config.get_path("models_dir", "models")
        outputs_dir = config.get_path("outputs_dir", "outputs")
        
        pretrained_model_path = resolve_path(f"{models_dir}/unet/flux1-dev-fp8.safetensors")
        clip_path = resolve_path(f"{models_dir}/clip/clip_l.safetensors")
        t5_path = resolve_path(f"{models_dir}/clip/t5xxl_fp8.safetensors")
        ae_path = resolve_path(f"{models_dir}/vae/ae.sft")
        output_dir = resolve_path(outputs_dir)

        # Add network weights parameter if existing adapter is provided
        network_weights_param = ""
        if network_weights:
            network_weights_param = f"--network_weights {resolve_path(network_weights)} {line_break}"

        ############# Optimizer args ########################
        if vram == "16G":
            # 16G VRAM
            optimizer = f"""--optimizer_type adafactor {line_break}
  --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" {line_break}
  --lr_scheduler constant_with_warmup {line_break}
  --max_grad_norm 0.0 {line_break}"""
        elif vram == "12G":
          # 12G VRAM
            optimizer = f"""--optimizer_type adafactor {line_break}
  --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" {line_break}
  --split_mode {line_break}
  --network_args "train_blocks=single" {line_break}
  --lr_scheduler constant_with_warmup {line_break}
  --max_grad_norm 0.0 {line_break}"""
        else:
            # 20G+ VRAM
            optimizer = f"--optimizer_type adamw8bit {line_break}"

        sh = f"""accelerate launch {line_break}
  --mixed_precision bf16 {line_break}
  --num_cpu_threads_per_process 1 {line_break}
  sd-scripts/flux_train_network.py {line_break}
  --pretrained_model_name_or_path {pretrained_model_path} {line_break}
  --clip_l {clip_path} {line_break}
  --t5xxl {t5_path} {line_break}
  --ae {ae_path} {line_break}
  --cache_latents_to_disk {line_break}
  --save_model_as safetensors {line_break}
  --sdpa --persistent_data_loader_workers {line_break}
  --max_data_loader_n_workers {workers} {line_break}
  --seed {seed} {line_break}
  --gradient_checkpointing {line_break}
  --mixed_precision bf16 {line_break}
  --save_precision bf16 {line_break}
  --network_module networks.lora_flux {line_break}
  --network_dim {network_dim} {line_break}
  {network_weights_param}{optimizer}{sample}
  --learning_rate {learning_rate} {line_break}
  --cache_text_encoder_outputs {line_break}
  --cache_text_encoder_outputs_to_disk {line_break}
  --fp8_base {line_break}
  --highvram {line_break}
  --max_train_epochs {max_train_epochs} {line_break}
  --save_every_n_epochs {save_every_n_epochs} {line_break}
  --dataset_config {resolve_path('dataset.toml')} {line_break}
  --output_dir {output_dir} {line_break}
  --output_name {output_name} {line_break}
  --timestep_sampling {timestep_sampling} {line_break}
  --discrete_flow_shift 3.1582 {line_break}
  --model_prediction_type raw {line_break}
  --guidance_scale {guidance_scale} {line_break}
  --loss_type l2 {line_break}"""
        return sh
    except Exception as e:
        logger.error(f"Error generating shell script: {e}")
        raise gr.Error(f"Error generating training script: {str(e)}")

def gen_toml(dataset_folder, resolution, class_tokens, num_repeats):
    """Generate TOML configuration for training."""
    try:
        toml = f"""[general]
shuffle_caption = false
caption_extension = '.txt'
keep_tokens = 1

[[datasets]]
resolution = {resolution}
batch_size = 1
keep_tokens = 1

  [[datasets.subsets]]
  image_dir = '{resolve_path_without_quotes(dataset_folder)}'
  class_tokens = '{class_tokens}'
  num_repeats = {num_repeats}"""
        return toml
    except Exception as e:
        logger.error(f"Error generating TOML config: {e}")
        raise gr.Error(f"Error generating dataset config: {str(e)}")

def update_total_steps(max_train_epochs, num_repeats, images):
    """Calculate and update total training steps."""
    try:
        num_images = len(images) if images else 0
        total_steps = max_train_epochs * num_images * num_repeats
        logger.info(f"Training steps calculation: epochs={max_train_epochs}, images={num_images}, repeats={num_repeats}, total={total_steps}")
        return gr.update(value=total_steps)
    except Exception as e:
        logger.error(f"Error updating total steps: {e}")
        return gr.update(value=0)

def get_samples():
    """Get sample images from outputs directory."""
    try:
        outputs_dir = config.get_path("outputs_dir", "outputs")
        samples_path = resolve_path_without_quotes(os.path.join(outputs_dir, 'sample'))
        
        if not os.path.exists(samples_path):
            return []
            
        files = [os.path.join(samples_path, file) for file in os.listdir(samples_path)]
        files.sort(key=lambda file: os.path.getctime(file), reverse=True)
        logger.info(f"Found {len(files)} sample files")
        return files
    except Exception as e:
        logger.error(f"Error getting samples: {e}")
        return []

def start_training(train_script, train_config, sample_prompts):
    """Start the training process."""
    try:
        # Create necessary directories
        models_dir = config.get_path("models_dir", "models")
        outputs_dir = config.get_path("outputs_dir", "outputs")
        
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(outputs_dir, exist_ok=True)

        file_type = "sh"
        if sys.platform == "win32":
            file_type = "bat"

        sh_filename = f"train.{file_type}"
        with open(sh_filename, 'w', encoding="utf-8") as file:
            file.write(train_script)
        gr.Info(f"Generated train script at {sh_filename}")

        with open('dataset.toml', 'w', encoding="utf-8") as file:
            file.write(train_config)
        gr.Info("Generated dataset.toml")

        with open('sample_prompts.txt', 'w', encoding='utf-8') as file:
            file.write(sample_prompts)
        gr.Info("Generated sample_prompts.txt")

        # Train
        if sys.platform == "win32":
            command = resolve_path_without_quotes('train.bat')
        else:
            command = f"bash {resolve_path('train.sh')}"

        # Run the training command
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        cwd = os.path.dirname(os.path.abspath(__file__))
        gr.Info("Started training")
        
        logger.info("Training process started")
        
        # Simple subprocess execution without real-time logging
        try:
            process = subprocess.Popen(
                command, 
                shell=True, 
                cwd=cwd, 
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            output_lines = []
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    output_lines.append(output.strip())
                    logger.info(output.strip())
                    # Yield progress update
                    yield "\n".join(output_lines[-50:])  # Show last 50 lines
            
            return_code = process.poll()
            if return_code == 0:
                gr.Info("Training Complete. Check the outputs folder for the LoRA files.", duration=None)
                logger.info("Training process completed successfully")
            else:
                gr.Error(f"Training failed with return code: {return_code}")
                logger.error(f"Training process failed with return code: {return_code}")
                
        except Exception as e:
            logger.error(f"Error running training command: {e}")
            gr.Error(f"Training failed: {str(e)}")
            yield f"Error: {str(e)}"
        
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        logger.error(traceback.format_exc())
        gr.Error(f"Error starting training: {str(e)}")

def update(
    lora_name,
    resolution,
    seed,
    workers,
    class_tokens,
    learning_rate,
    network_dim,
    max_train_epochs,
    save_every_n_epochs,
    timestep_sampling,
    guidance_scale,
    vram,
    num_repeats,
    sample_prompts,
    sample_every_n_steps,
    pretrained_adapter,
):
    """Update training configuration."""
    try:
        output_name = slugify(lora_name)
        datasets_dir = config.get_path("datasets_dir", "datasets")
        dataset_folder = os.path.join(datasets_dir, output_name)
        
        # Set network_weights if pretrained adapter is selected
        network_weights = None
        if pretrained_adapter and pretrained_adapter != "None":
            network_weights = os.path.join(PRETRAINED_ADAPTERS_DIR, pretrained_adapter)
        
        sh = gen_sh(
            output_name,
            resolution,
            seed,
            workers,
            learning_rate,
            network_dim,
            max_train_epochs,
            save_every_n_epochs,
            timestep_sampling,
            guidance_scale,
            vram,
            sample_prompts,
            sample_every_n_steps,
            network_weights,
        )
        toml = gen_toml(
            dataset_folder,
            resolution,
            class_tokens,
            num_repeats
        )
        return gr.update(value=sh), gr.update(value=toml), dataset_folder
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        raise gr.Error(f"Error updating configuration: {str(e)}")

def loaded():
    """Called when the interface is loaded."""
    logger.info("Application launched successfully")

def update_sample(concept_sentence):
    """Update sample prompts with concept sentence."""
    return gr.update(value=concept_sentence)

def detect_existing_images(lora_name, concept_sentence):
    """Detect and load existing images from dataset folder."""
    try:
        if not lora_name:
            raise gr.Error("Please enter a LoRA name first")
        
        # Get dataset folder path
        datasets_dir = config.get_path("datasets_dir", "datasets")
        dataset_folder = os.path.join(datasets_dir, slugify(lora_name))
        
        if not os.path.exists(dataset_folder):
            raise gr.Error(f"Dataset folder not found: {dataset_folder}")
        
        # Find existing images
        existing_images = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        
        for file in os.listdir(dataset_folder):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                existing_images.append(os.path.join(dataset_folder, file))
        
        if not existing_images:
            raise gr.Error(f"No images found in {dataset_folder}")
        
        if len(existing_images) > MAX_IMAGES:
            existing_images = existing_images[:MAX_IMAGES]
            gr.Warning(f"Found {len(os.listdir(dataset_folder))} images, showing first {MAX_IMAGES}")
        
        gr.Info(f"Found {len(existing_images)} images in dataset folder!")
        
        # Call load_captioning with the detected images
        return load_captioning(existing_images, concept_sentence)
        
    except Exception as e:
        logger.error(f"Error detecting existing images: {e}")
        raise gr.Error(f"Error detecting images: {str(e)}")

def get_available_adapters():
    """Get list of available pretrained adapters."""
    adapters = []
    if os.path.exists(PRETRAINED_ADAPTERS_DIR):
        for file in os.listdir(PRETRAINED_ADAPTERS_DIR):
            if file.endswith('.safetensors'):
                adapters.append(file)
    return sorted(adapters)

def upload_pretrained_adapter(files):
    """Handle uploaded pretrained adapter files."""
    if not files:
        return gr.update(choices=get_available_adapters(), value=None), "No files uploaded"
    
    uploaded_count = 0
    for file in files:
        if file.name.endswith('.safetensors'):
            try:
                # Copy to pretrained adapters directory
                dest_path = os.path.join(PRETRAINED_ADAPTERS_DIR, os.path.basename(file.name))
                shutil.copy2(file.name, dest_path)
                uploaded_count += 1
                logger.info(f"Uploaded adapter: {os.path.basename(file.name)}")
            except Exception as e:
                logger.error(f"Error uploading {file.name}: {e}")
    
    message = f"✅ Successfully uploaded {uploaded_count} adapter(s)" if uploaded_count > 0 else "❌ No valid .safetensors files found"
    return gr.update(choices=get_available_adapters(), value=None), message

def delete_adapter(adapter_name):
    """Delete a pretrained adapter."""
    if not adapter_name:
        return gr.update(choices=get_available_adapters(), value=None), "No adapter selected"
    
    try:
        adapter_path = os.path.join(PRETRAINED_ADAPTERS_DIR, adapter_name)
        if os.path.exists(adapter_path):
            os.remove(adapter_path)
            logger.info(f"Deleted adapter: {adapter_name}")
            return gr.update(choices=get_available_adapters(), value=None), f"✅ Deleted: {adapter_name}"
        else:
            return gr.update(choices=get_available_adapters(), value=None), f"❌ Adapter not found: {adapter_name}"
    except Exception as e:
        logger.error(f"Error deleting adapter {adapter_name}: {e}")
        return gr.update(choices=get_available_adapters(), value=None), f"❌ Error deleting: {str(e)}"

def get_adapter_info(adapter_name):
    """Get information about a selected adapter."""
    if not adapter_name:
        return "No adapter selected"
    
    adapter_path = os.path.join(PRETRAINED_ADAPTERS_DIR, adapter_name)
    if os.path.exists(adapter_path):
        try:
            file_size = os.path.getsize(adapter_path)
            file_size_mb = file_size / (1024 * 1024)
            mod_time = os.path.getmtime(adapter_path)
            mod_time_str = gr.utils.format_datetime(mod_time)
            
            return f"""📁 **Adapter Info:**
**File:** {adapter_name}
**Size:** {file_size_mb:.2f} MB
**Modified:** {mod_time_str}
**Path:** `{adapter_path}`

💡 **Training Mode:** Continue training from this adapter
⚠️ **Note:** Make sure this adapter is compatible with FLUX"""
        except Exception as e:
            return f"❌ Error reading adapter info: {str(e)}"
    else:
        return "❌ Adapter file not found"

theme = gr.themes.Monochrome(
    text_size=gr.themes.Size(lg="18px", md="15px", sm="13px", xl="22px", xs="12px", xxl="24px", xxs="9px"),
    font=[gr.themes.GoogleFont("Source Sans Pro"), "ui-sans-serif", "system-ui", "sans-serif"],
)
css = """
@keyframes rotate {
    0% {
        transform: rotate(0deg);
    }
    100% {
        transform: rotate(360deg);
    }
}
h1{font-family: georgia; font-style: italic; font-weight: bold; font-size: 30px; letter-spacing: -1px;}
h3{margin-top: 0}
.tabitem{border: 0px}
.group_padding{}
nav{position: fixed; top: 0; left: 0; right: 0; z-index: 1000; text-align: center; padding: 10px; box-sizing: border-box; display: flex; align-items: center; backdrop-filter: blur(10px); }
nav button { background: none; color: firebrick; font-weight: bold; border: 2px solid firebrick; padding: 5px 10px; border-radius: 5px; font-size: 14px; }
nav img { height: 40px; width: 40px; border-radius: 40px; }
nav img.rotate { animation: rotate 2s linear infinite; }
.flexible { flex-grow: 1; }
.tast-details { margin: 10px 0 !important; }
.toast-wrap { bottom: var(--size-4) !important; top: auto !important; border: none !important; backdrop-filter: blur(10px); }
.toast-title, .toast-text, .toast-icon, .toast-close { color: black !important; font-size: 14px; }
.toast-body { border: none !important; }
#terminal { box-shadow: none !important; margin-bottom: 25px; background: rgba(0,0,0,0.03); }
#terminal .generating { border: none !important; }
#terminal label { position: absolute !important; }
#container { margin-top: 50px; }
.hidden { display: none !important; }
.codemirror-wrapper .cm-line { font-size: 12px !important; }
"""

js = """
function() {
    let autoscroll = document.querySelector("#autoscroll")
    if (window.iidxx) {
        window.clearInterval(window.iidxx);
    }
    window.iidxx = window.setInterval(function() {
        let text=document.querySelector(".codemirror-wrapper .cm-line").innerText.trim()
        let img = document.querySelector("#logo")
        if (text.length > 0) {
            autoscroll.classList.remove("hidden")
            if (autoscroll.classList.contains("on")) {
                autoscroll.textContent = "Autoscroll ON"
                window.scrollTo(0, document.body.scrollHeight, { behavior: "smooth" });
                img.classList.add("rotate")
            } else {
                autoscroll.textContent = "Autoscroll OFF"
                img.classList.remove("rotate")
            }
        }
    }, 500);
    console.log("autoscroll", autoscroll)
    autoscroll.addEventListener("click", (e) => {
        autoscroll.classList.toggle("on")
    })
}
"""

with gr.Blocks(elem_id="app", theme=theme, css=css, fill_width=True) as demo:
    output_components = []
    with gr.Row():
        gr.HTML("""<nav>
    <img id='logo' src='/file=icon.png' width='80' height='80'>
    <div class='flexible'></div>
    <button id='autoscroll' class='on hidden'></button>
</nav>
""")
    with gr.Row(elem_id='container'):
        with gr.Column():
            gr.Markdown(
                """# Step 1. LoRA Info
<p style="margin-top:0">Configure your LoRA train settings.</p>
""", elem_classes="group_padding")
            lora_name = gr.Textbox(
                label="The name of your LoRA",
                info="This has to be a unique name",
                placeholder="e.g.: Persian Miniature Painting style, Cat Toy",
            )
            concept_sentence = gr.Textbox(
                label="Trigger word/sentence",
                info="Trigger word or sentence to be used",
                placeholder="uncommon word like p3rs0n or trtcrd, or sentence like 'in the style of CNSTLL'",
                interactive=True,
            )
            vram = gr.Radio(["20G", "16G", "12G" ], value="20G", label="VRAM", interactive=True)
            num_repeats = gr.Number(value=10, precision=0, label="Repeat trains per image", interactive=True)
            max_train_epochs = gr.Number(label="Max Train Epochs", value=16, interactive=True)
            total_steps = gr.Number(0, interactive=False, label="Expected training steps")
            sample_prompts = gr.Textbox("", lines=5, label="Sample Image Prompts (Separate with new lines)", interactive=True)
            sample_every_n_steps = gr.Number(0, precision=0, label="Sample Image Every N Steps", interactive=True)
            with gr.Accordion("Advanced options", open=False):
                #resolution = gr.Number(label="Resolution", value=512, minimum=512, maximum=1024, step=512)
                seed = gr.Number(label="Seed", value=42, interactive=True)
                workers = gr.Number(label="Workers", value=2, interactive=True)
                learning_rate = gr.Textbox(label="Learning Rate", value="8e-4", interactive=True)
                #learning_rate = gr.Number(label="Learning Rate", value=4e-4, minimum=1e-6, maximum=1e-3, step=1e-6)

                save_every_n_epochs = gr.Number(label="Save every N epochs", value=4, interactive=True)

                guidance_scale = gr.Number(label="Guidance Scale", value=1.0, interactive=True)

                timestep_sampling = gr.Textbox(label="Timestep Sampling", value="shift", interactive=True)

    #            steps = gr.Number(label="Steps", value=1000, minimum=1, maximum=10000, step=1)
                network_dim = gr.Number(label="LoRA Rank", value=4, minimum=4, maximum=128, step=4, interactive=True)
                resolution = gr.Number(value=512, precision=0, label="Resize dataset images")
            
            # Pretrained Adapter Section
            gr.Markdown(
                """## 🔄 Pretrained Adapter (Optional)
<p style="margin-top:0">Continue training from an existing LoRA adapter.</p>
""", elem_classes="group_padding")
            
            with gr.Group():
                with gr.Row():
                    pretrained_adapter = gr.Dropdown(
                        choices=["None"] + get_available_adapters(),
                        value="None",
                        label="Select Existing Adapter",
                        info="Choose an adapter to continue training from",
                        interactive=True,
                        scale=2
                    )
                    refresh_adapters = gr.Button("🔄 Refresh", variant="secondary", scale=1)
                
                with gr.Row():
                    upload_adapter = gr.File(
                        file_types=[".safetensors"],
                        label="Upload New Adapter (.safetensors)",
                        file_count="multiple",
                        interactive=True,
                        scale=2
                    )
                    delete_adapter_btn = gr.Button("🗑️ Delete Selected", variant="stop", scale=1)
                
                adapter_info = gr.Markdown("No adapter selected", elem_classes="info-message")
                upload_status = gr.Textbox(label="Upload Status", interactive=False, visible=False)
                
                gr.Markdown("""
**💡 How to use:**
- **Upload**: Drop your `.safetensors` LoRA files above
- **Select**: Choose an adapter from the dropdown
- **Train**: The selected adapter will be used as starting point
- **Folder**: You can also place adapters in `pretrained_adapters/` folder

**⚠️ Note:** Make sure the adapter is compatible with FLUX and uses similar settings
""", elem_classes="info-message")
        with gr.Column():
            gr.Markdown(
                """# Step 2. Dataset
<p style="margin-top:0">Make sure the captions include the trigger word.</p>
""", elem_classes="group_padding")
            with gr.Group():
                images = gr.File(
                    file_types=["image", ".txt"],
                    label="Upload your images (or place directly in datasets folder)",
                    file_count="multiple",
                    interactive=True,
                    visible=True,
                    scale=1,
                )
                gr.Markdown("""
**💡 For 300+ images:** Instead of uploading here, copy your images directly to:
`datasets/[your-lora-name]/` folder for much faster processing.
""", elem_classes="info-message")
                
                # Add button to detect existing images
                detect_existing = gr.Button("🔍 Detect Existing Images in Dataset Folder", variant="secondary")
                
            with gr.Group(visible=False) as captioning_area:
                do_captioning = gr.Button("Add AI captions with Gemini")
                output_components.append(captioning_area)
                #output_components = [captioning_area]
                caption_list = []
                for i in range(1, MAX_IMAGES + 1):
                    locals()[f"captioning_row_{i}"] = gr.Row(visible=False)
                    with locals()[f"captioning_row_{i}"]:
                        locals()[f"image_{i}"] = gr.Image(
                            type="filepath",
                            width=111,
                            height=111,
                            min_width=111,
                            interactive=False,
                            scale=2,
                            show_label=False,
                            show_share_button=False,
                            show_download_button=False,
                        )
                        locals()[f"caption_{i}"] = gr.Textbox(
                            label=f"Caption {i}", scale=15, interactive=True
                        )

                    output_components.append(locals()[f"captioning_row_{i}"])
                    output_components.append(locals()[f"image_{i}"])
                    output_components.append(locals()[f"caption_{i}"])
                    caption_list.append(locals()[f"caption_{i}"])
        with gr.Column():
            gr.Markdown(
                """# Step 3. Train
<p style="margin-top:0">Press start to start training.</p>
""", elem_classes="group_padding")
            start = gr.Button("Start training", visible=False)
            output_components.append(start)
            train_script = gr.Textbox(label="Train script", max_lines=100, interactive=True)
            train_config = gr.Textbox(label="Train config", max_lines=100, interactive=True)
    with gr.Row():
        terminal = gr.Textbox(
            label="Train log", 
            elem_id="terminal",
            lines=15,
            max_lines=15,
            interactive=False,
            autoscroll=True,
            show_copy_button=True
        )
    with gr.Row():
        gallery = gr.Gallery(get_samples, label="Samples", every=10, columns=6)


    dataset_folder = gr.State()

    listeners = [
        lora_name,
        resolution,
        seed,
        workers,
        concept_sentence,
        learning_rate,
        network_dim,
        max_train_epochs,
        save_every_n_epochs,
        timestep_sampling,
        guidance_scale,
        vram,
        num_repeats,
        sample_prompts,
        sample_every_n_steps,
        pretrained_adapter,
    ]


    for listener in listeners:
        listener.change(update, inputs=listeners, outputs=[train_script, train_config, dataset_folder])

    images.upload(
        load_captioning,
        inputs=[images, concept_sentence],
        outputs=output_components
    )

    images.delete(
        load_captioning,
        inputs=[images, concept_sentence],
        outputs=output_components
    )

    images.clear(
        hide_captioning,
        outputs=[captioning_area, start]
    )


    # update total steps

    max_train_epochs.change(
        fn=update_total_steps,
        inputs=[max_train_epochs, num_repeats, images],
        outputs=[total_steps]
    )
    num_repeats.change(
        fn=update_total_steps,
        inputs=[max_train_epochs, num_repeats, images],
        outputs=[total_steps]
    )

    images.upload(
        fn=update_total_steps,
        inputs=[max_train_epochs, num_repeats, images],
        outputs=[total_steps]
    )
    images.delete(
        fn=update_total_steps,
        inputs=[max_train_epochs, num_repeats, images],
        outputs=[total_steps]
    )
    images.clear(
        fn=update_total_steps,
        inputs=[max_train_epochs, num_repeats, images],
        outputs=[total_steps]
    )


    concept_sentence.change(fn=update_sample, inputs=[concept_sentence], outputs=sample_prompts)

    start.click(fn=create_dataset, inputs=[dataset_folder, resolution, images] + caption_list, outputs=dataset_folder).then(
        fn=start_training,
        inputs=[
            train_script,
            train_config,
            sample_prompts,
        ],
        outputs=terminal,
    )

    do_captioning.click(fn=run_captioning, inputs=[images, concept_sentence] + caption_list, outputs=caption_list)
    
    # Add detect existing images button handler
    detect_existing.click(
        fn=detect_existing_images, 
        inputs=[lora_name, concept_sentence], 
        outputs=output_components
    )
    
    # Pretrained Adapter Event Handlers
    upload_adapter.upload(
        fn=upload_pretrained_adapter,
        inputs=[upload_adapter],
        outputs=[pretrained_adapter, upload_status]
    )
    
    refresh_adapters.click(
        fn=lambda: gr.update(choices=["None"] + get_available_adapters()),
        outputs=[pretrained_adapter]
    )
    
    delete_adapter_btn.click(
        fn=delete_adapter,
        inputs=[pretrained_adapter],
        outputs=[pretrained_adapter, upload_status]
    )
    
    pretrained_adapter.change(
        fn=get_adapter_info,
        inputs=[pretrained_adapter],
        outputs=[adapter_info]
    )

    demo.load(fn=loaded, js=js)

if __name__ == "__main__":
    cwd = os.path.dirname(os.path.abspath(__file__))
    demo.launch(show_error=True, allowed_paths=[cwd], share=True)
