"""
Frame Generator Module
Handles core diffusion model operations for generating individual frames.
"""

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from PIL import Image
from typing import Optional, List
import numpy as np


class FrameGenerator:
    """
    Generates individual frames using Stable Diffusion.
    Supports both text-to-image and image-to-image generation.
    """
    
    def __init__(
        self,
        model_id: str = "Manojb/stable-diffusion-2-1-base",
        device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        torch_dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    ):
        """
        Initialize the frame generator.
        
        Args:
            model_id: Hugging Face model identifier
            device: Device to run the model on (cuda/cpu)
            torch_dtype: Data type for model weights
        """
        self.device = device
        self.torch_dtype = torch_dtype
        
        print(f"Loading model: {model_id}")
        print(f"Device: {device}")
        
        # Load text-to-image pipeline
        self.txt2img_pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            safety_checker=None  # Disable for faster inference
        ).to(device)
        
        # Load image-to-image pipeline
        self.img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            safety_checker=None
        ).to(device)
        
        # Enable memory optimizations
        if device in ("cuda", "mps"):
            self.txt2img_pipe.enable_attention_slicing()
            self.img2img_pipe.enable_attention_slicing()
        
        print("Model loaded successfully!")
    
    def generate_first_frame(
        self,
        prompt: str,
        negative_prompt: str = "blurry, bad quality, distorted",
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> Image.Image:
        """
        Generate the first frame from a text prompt.
        
        Args:
            prompt: Text description of the desired image
            negative_prompt: What to avoid in the generation
            width: Image width
            height: Image height
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt
            seed: Random seed for reproducibility
            
        Returns:
            PIL Image of the generated frame
        """
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)
        
        output = self.txt2img_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )
        
        return output.images[0]
    
    def generate_next_frame(
        self,
        previous_frame: Image.Image,
        prompt: str,
        negative_prompt: str = "blurry, bad quality, distorted",
        strength: float = 0.5,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> Image.Image:
        """
        Generate the next frame using image-to-image with previous frame.
        
        Args:
            previous_frame: Previous frame to condition on
            prompt: Text description (can be modified for motion)
            negative_prompt: What to avoid
            strength: How much to change from previous frame (0.0-1.0)
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt
            seed: Random seed
            
        Returns:
            PIL Image of the generated frame
        """
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)
        
        output = self.img2img_pipe(
            prompt=prompt,
            image=previous_frame,
            negative_prompt=negative_prompt,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )
        
        return output.images[0]
    
    def generate_frame_sequence(
        self,
        base_prompt: str,
        num_frames: int = 16,
        strength: float = 0.5,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        progress_callback: Optional[callable] = None
    ) -> List[Image.Image]:
        """
        Generate a sequence of frames with basic img2img.
        
        Args:
            base_prompt: Base text prompt for the sequence
            num_frames: Number of frames to generate
            strength: Variation strength for img2img
            guidance_scale: Prompt adherence
            seed: Random seed for reproducibility
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of PIL Images
        """
        frames = []
        
        # Generate first frame
        if progress_callback:
            progress_callback(0, num_frames, "Generating first frame...")
        
        first_frame = self.generate_first_frame(
            prompt=base_prompt,
            guidance_scale=guidance_scale,
            seed=seed
        )
        frames.append(first_frame)
        
        # Generate subsequent frames
        for i in range(1, num_frames):
            if progress_callback:
                progress_callback(i, num_frames, f"Generating frame {i+1}/{num_frames}...")
            
            next_frame = self.generate_next_frame(
                previous_frame=frames[-1],
                prompt=base_prompt,
                strength=strength,
                guidance_scale=guidance_scale,
                seed=seed + i if seed else None
            )
            frames.append(next_frame)
        
        if progress_callback:
            progress_callback(num_frames, num_frames, "Complete!")
        
        return frames


if __name__ == "__main__":
    # Test the frame generator
    generator = FrameGenerator()
    
    print("\nGenerating test animation...")
    frames = generator.generate_frame_sequence(
        base_prompt="a red ball on a white background",
        num_frames=5,
        seed=42
    )
    
    print(f"Generated {len(frames)} frames")
    print("Frame generator test complete!")

