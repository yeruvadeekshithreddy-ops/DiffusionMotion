"""
Interpolation Module
Handles latent space interpolation for smooth frame transitions.
"""

import torch
from diffusers import AutoencoderKL, StableDiffusionPipeline
from PIL import Image
from typing import List, Optional
import numpy as np


class LatentInterpolator:
    """
    Interpolates between keyframes in latent space for smooth animations.
    """
    
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-2-1-base",
        device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        torch_dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    ):
        """
        Initialize the interpolator.
        
        Args:
            model_id: Stable Diffusion model for VAE
            device: Device to run on
            torch_dtype: Data type for tensors
        """
        self.device = device
        self.torch_dtype = torch_dtype
        
        print(f"Loading VAE from: {model_id}")
        
        # Load VAE for encoding/decoding
        self.vae = AutoencoderKL.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=torch_dtype
        ).to(device)
        
        print("VAE loaded successfully!")
    
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """
        Encode PIL Image to latent representation.
        
        Args:
            image: PIL Image to encode
            
        Returns:
            Latent tensor
        """
        # Preprocess image
        image = image.resize((512, 512))
        image_np = np.array(image).astype(np.float32) / 255.0
        image_np = image_np * 2.0 - 1.0  # Normalize to [-1, 1]
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(device=self.device, dtype=self.torch_dtype)
        
        # Encode to latent space
        with torch.no_grad():
            latent = self.vae.encode(image_tensor).latent_dist.sample()
            latent = latent * 0.18215  # Scaling factor for SD
        
        return latent
    
    def decode_latent(self, latent: torch.Tensor) -> Image.Image:
        """
        Decode latent representation to PIL Image.
        
        Args:
            latent: Latent tensor
            
        Returns:
            PIL Image
        """
        # Decode from latent space
        with torch.no_grad():
            latent = latent / 0.18215
            image_tensor = self.vae.decode(latent).sample
        
        # Post-process
        image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
        image_np = image_tensor.cpu().permute(0, 2, 3, 1).numpy()[0]
        image_np = (image_np * 255).astype(np.uint8)
        
        return Image.fromarray(image_np)
    
    def interpolate_latents(
        self,
        latent_start: torch.Tensor,
        latent_end: torch.Tensor,
        num_steps: int = 10,
        interpolation_type: str = "linear"
    ) -> List[torch.Tensor]:
        """
        Interpolate between two latent representations.
        
        Args:
            latent_start: Starting latent
            latent_end: Ending latent
            num_steps: Number of interpolation steps
            interpolation_type: Type of interpolation (linear, slerp)
            
        Returns:
            List of interpolated latents
        """
        interpolated = []
        
        for i in range(num_steps):
            alpha = i / (num_steps - 1) if num_steps > 1 else 0
            
            if interpolation_type == "linear":
                # Linear interpolation
                latent = (1 - alpha) * latent_start + alpha * latent_end
            elif interpolation_type == "slerp":
                # Spherical linear interpolation (better for latent spaces)
                latent = self._slerp(latent_start, latent_end, alpha)
            else:
                raise ValueError(f"Unknown interpolation type: {interpolation_type}")
            
            interpolated.append(latent)
        
        return interpolated
    
    def _slerp(
        self,
        v0: torch.Tensor,
        v1: torch.Tensor,
        t: float,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Spherical linear interpolation.
        
        Args:
            v0: Start vector
            v1: End vector
            t: Interpolation parameter [0, 1]
            eps: Small value to prevent division by zero
            
        Returns:
            Interpolated tensor
        """
        # Normalize vectors
        v0_norm = v0 / (torch.norm(v0) + eps)
        v1_norm = v1 / (torch.norm(v1) + eps)
        
        # Compute angle
        dot = (v0_norm * v1_norm).sum()
        dot = torch.clamp(dot, -1.0, 1.0)
        omega = torch.acos(dot)
        
        # Compute interpolation
        so = torch.sin(omega)
        if so < eps:
            # Vectors are nearly parallel, use linear interpolation
            return (1.0 - t) * v0 + t * v1
        
        return (torch.sin((1.0 - t) * omega) / so) * v0 + (torch.sin(t * omega) / so) * v1
    
    def generate_interpolated_sequence(
        self,
        keyframes: List[Image.Image],
        frames_between: int = 4,
        interpolation_type: str = "slerp"
    ) -> List[Image.Image]:
        """
        Generate smooth sequence by interpolating between keyframes.
        
        Args:
            keyframes: List of keyframe images
            frames_between: Number of frames to generate between each keyframe
            interpolation_type: Interpolation method
            
        Returns:
            List of all frames (keyframes + interpolated)
        """
        if len(keyframes) < 2:
            return keyframes
        
        all_frames = []
        
        for i in range(len(keyframes) - 1):
            # Encode keyframes to latent space
            latent_start = self.encode_image(keyframes[i])
            latent_end = self.encode_image(keyframes[i + 1])
            
            # Interpolate
            interpolated_latents = self.interpolate_latents(
                latent_start,
                latent_end,
                num_steps=frames_between + 2,  # +2 to include start and end
                interpolation_type=interpolation_type
            )
            
            # Decode all except the last (avoid duplicates)
            for latent in interpolated_latents[:-1]:
                frame = self.decode_latent(latent)
                all_frames.append(frame)
        
        # Add final keyframe
        all_frames.append(keyframes[-1])
        
        return all_frames


if __name__ == "__main__":
    print("Latent interpolation module loaded.")
    print("Use with keyframe images to generate smooth interpolations.")

