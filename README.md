# 🎬 DiffusionMotion - AI-Assisted Animation

AI-powered frame-by-frame animation generator using Stable Diffusion and ControlNet for temporal consistency.

## 🎯 Project Goal

Generate smooth, temporally-consistent animated sequences from text prompts using diffusion models.

## ✨ Features

- 🖼️ Text-to-animation generation
- 🎨 ControlNet for temporal consistency
- 🔄 Latent space interpolation for smooth transitions
- 🎥 Export to GIF/MP4
- 🌐 Interactive Gradio web interface

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Application

```bash
python app.py
```

### 3. Generate Your First Animation

Open the Gradio interface and enter a prompt like:
- "A red ball bouncing up and down"
- "A blue cube rotating slowly"
- "A stick figure walking from left to right"

## 📁 Project Structure

```
DiffusionMotion/
├── src/
│   ├── frame_generator.py      # Core diffusion frame generation
│   ├── controlnet_pipeline.py  # ControlNet for consistency
│   ├── interpolation.py        # Latent space interpolation
│   └── animation_builder.py    # Frames → GIF/MP4
├── models/                      # Downloaded model weights (auto-downloaded)
├── outputs/                     # Generated animations
├── app.py                       # Gradio UI
├── requirements.txt
└── README.md
```

## 🛠️ Technical Details

### Models Used
- **Stable Diffusion 2.1**: Base image generation
- **ControlNet-Canny**: Edge-guided temporal consistency
- **VAE**: Latent space encoding/decoding

### Key Concepts
- **Latent Diffusion**: Generate images in compressed latent space
- **Img2Img**: Condition new frames on previous frames
- **ControlNet**: Use structural guides to maintain shape consistency
- **Temporal Coherence**: Prevent frame-to-frame flickering

## 📊 Parameters

- **num_frames**: Number of animation frames (default: 16)
- **guidance_scale**: Prompt adherence strength (default: 7.5)
- **strength**: Img2img variation strength (default: 0.5)
- **fps**: Animation playback speed (default: 8)

## 🎓 Learning Outcomes

- ✅ Latent diffusion models
- ✅ Image-to-image generation
- ✅ Noise schedules and denoising
- ✅ Temporal coherence techniques
- ✅ ControlNet conditioning

## 📝 License

MIT License

## 🙏 Credits

Built with:
- [Stable Diffusion](https://github.com/Stability-AI/stablediffusion)
- [Diffusers](https://github.com/huggingface/diffusers)
- [ControlNet](https://github.com/lllyasviel/ControlNet)

