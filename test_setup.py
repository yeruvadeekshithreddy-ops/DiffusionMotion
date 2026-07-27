"""
Setup Verification Script
Tests that all dependencies are installed and working correctly.
"""

import sys
from pathlib import Path

print("=" * 70)
print("🔍 DiffusionMotion - Setup Verification")
print("=" * 70)
print()

# Test 1: Python Version
print("1️⃣  Checking Python version...")
python_version = sys.version_info
print(f"   Python {python_version.major}.{python_version.minor}.{python_version.micro}")
if python_version.major >= 3 and python_version.minor >= 10:
    print("   ✅ Python version OK")
else:
    print("   ⚠️  Warning: Python 3.10+ recommended")
print()

# Test 2: Core Dependencies
print("2️⃣  Checking core dependencies...")
dependencies = {
    'torch': 'PyTorch',
    'diffusers': 'Diffusers',
    'transformers': 'Transformers',
    'PIL': 'Pillow',
    'gradio': 'Gradio',
    'imageio': 'ImageIO',
    'cv2': 'OpenCV',
    'numpy': 'NumPy'
}

all_installed = True
for module_name, display_name in dependencies.items():
    try:
        __import__(module_name)
        print(f"   ✅ {display_name}")
    except ImportError:
        print(f"   ❌ {display_name} - NOT INSTALLED")
        all_installed = False

print()

# Test 3: PyTorch & CUDA
print("3️⃣  Checking PyTorch and CUDA...")
try:
    import torch
    print(f"   PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"   ✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("   ⚠️  CUDA not available - will run on CPU (slower)")
        print("   Tip: Install PyTorch with CUDA support for faster generation")
except Exception as e:
    print(f"   ❌ Error: {e}")

print()

# Test 4: Project Structure
print("4️⃣  Checking project structure...")
required_paths = [
    'src',
    'src/frame_generator.py',
    'src/controlnet_pipeline.py',
    'src/interpolation.py',
    'src/animation_builder.py',
    'app.py',
    'requirements.txt',
    'README.md'
]

for path_str in required_paths:
    path = Path(path_str)
    if path.exists():
        print(f"   ✅ {path_str}")
    else:
        print(f"   ❌ {path_str} - MISSING")
        all_installed = False

print()

# Test 5: Import Project Modules
print("5️⃣  Testing project imports...")
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.frame_generator import FrameGenerator
    print("   ✅ FrameGenerator")
except Exception as e:
    print(f"   ❌ FrameGenerator: {e}")
    all_installed = False

try:
    from src.animation_builder import AnimationBuilder
    print("   ✅ AnimationBuilder")
except Exception as e:
    print(f"   ❌ AnimationBuilder: {e}")
    all_installed = False

try:
    from src.interpolation import LatentInterpolator
    print("   ✅ LatentInterpolator")
except Exception as e:
    print(f"   ❌ LatentInterpolator: {e}")
    all_installed = False

try:
    from src.controlnet_pipeline import ControlNetFrameGenerator
    print("   ✅ ControlNetFrameGenerator")
except Exception as e:
    print(f"   ❌ ControlNetFrameGenerator: {e}")
    all_installed = False

print()

# Test 6: Output Directories
print("6️⃣  Checking output directories...")
output_dirs = ['outputs', 'models']
for dir_name in output_dirs:
    dir_path = Path(dir_name)
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"   📁 Created: {dir_name}/")
    else:
        print(f"   ✅ {dir_name}/")

print()

# Summary
print("=" * 70)
if all_installed:
    print("✅ Setup verification PASSED!")
    print()
    print("🚀 Next Steps:")
    print("   1. Run: python app.py")
    print("   2. Open browser: http://127.0.0.1:7860")
    print("   3. Start generating animations!")
    print()
    print("⏱️  Note: First run will download ~5GB of models (one-time only)")
else:
    print("❌ Setup verification FAILED!")
    print()
    print("🔧 Fix issues above, then run:")
    print("   pip install -r requirements.txt")
    print()
    print("💡 Check SETUP.md for detailed installation guide")

print("=" * 70)

