# 🚀 Setup Guide for DiffusionMotion

## Prerequisites

- Python 3.10 or higher
- NVIDIA GPU with CUDA support (optional but recommended)
- ~10GB free disk space for models
- ~4GB VRAM for GPU inference (can run on CPU but slower)

## Installation Steps

### 1. Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 3. Install PyTorch

**For CUDA (GPU) - Windows/Linux:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**For CPU only:**
```bash
pip install torch torchvision
```

**For Mac:**
```bash
pip install torch torchvision
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- Diffusers (Hugging Face diffusion models)
- Transformers (text encoding)
- ControlNet auxiliary models
- Gradio (web interface)
- Image processing libraries
- And more...

### 5. Verify Installation

Run the test script:
```bash
python test_setup.py
```

This will:
- Check all dependencies
- Verify GPU/CPU availability
- Test model loading (small test)

## First Run

### Option 1: Gradio Web Interface (Recommended)

```bash
python app.py
```

Then open your browser to: `http://127.0.0.1:7860`

**Note:** First run will download ~5GB of model weights (Stable Diffusion 2.1). This only happens once!

### Option 2: Direct Python Script

```python
from src.frame_generator import FrameGenerator
from src.animation_builder import AnimationBuilder

# Initialize
generator = FrameGenerator()
builder = AnimationBuilder()

# Generate frames
frames = generator.generate_frame_sequence(
    base_prompt="A red ball bouncing",
    num_frames=10,
    seed=42
)

# Create GIF
builder.create_gif(frames, "my_animation.gif", fps=8)
```

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution:**
- Reduce image resolution (edit models to 256x256)
- Use fewer frames
- Enable model offloading: `pipe.enable_model_cpu_offload()`

### Issue: Slow Generation

**Solutions:**
- Use GPU if available
- Reduce `num_inference_steps` (default: 30 → try 20)
- Reduce number of frames
- Use lower resolution

### Issue: Model Download Fails

**Solution:**
```bash
# Set HuggingFace cache directory
export HF_HOME="./models"  # Linux/Mac
set HF_HOME=./models        # Windows CMD
$env:HF_HOME="./models"     # Windows PowerShell
```

### Issue: Import Errors

**Solution:**
Make sure virtual environment is activated:
```bash
# Check Python location
which python   # Linux/Mac
where python   # Windows

# Should point to venv/bin/python or venv\Scripts\python.exe
```

## Performance Expectations

### GPU (NVIDIA RTX 3060 or better):
- Single frame: ~2-3 seconds
- 16-frame animation: ~40-60 seconds

### CPU (Modern i7/i9):
- Single frame: ~30-60 seconds
- 16-frame animation: ~10-15 minutes

## Next Steps

1. ✅ Run `python app.py` to start the web interface
2. ✅ Try example prompts to understand the system
3. ✅ Experiment with different parameters
4. ✅ Check out the code in `src/` to understand how it works
5. ✅ Try implementing ControlNet for better consistency (Phase 3)

## Advanced: Using ControlNet

ControlNet is not enabled by default. To use it:

```python
from src.controlnet_pipeline import ControlNetFrameGenerator

# Initialize (first run will download ControlNet weights)
controlnet_gen = ControlNetFrameGenerator()

# Generate with edge guidance
# (See controlnet_pipeline.py for full API)
```

## Resources

- [Stable Diffusion Docs](https://huggingface.co/docs/diffusers/using-diffusers/sdxl)
- [ControlNet Paper](https://arxiv.org/abs/2302.05543)
- [Diffusers Library](https://github.com/huggingface/diffusers)

## Questions?

Check the main `README.md` for project overview and architecture details.

Happy animating! 🎬✨

