"""
DiffusionMotion - AI-Assisted Animation
Frame-by-frame animation generation using diffusion models.
"""

from .frame_generator import FrameGenerator
from .controlnet_pipeline import ControlNetFrameGenerator
from .interpolation import LatentInterpolator
from .animation_builder import AnimationBuilder

__all__ = [
    'FrameGenerator',
    'ControlNetFrameGenerator',
    'LatentInterpolator',
    'AnimationBuilder'
]

__version__ = '0.1.0'

