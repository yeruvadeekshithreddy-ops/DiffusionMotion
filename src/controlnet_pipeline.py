"""
ControlNet Pipeline Module
Handles ControlNet-based frame generation for improved temporal consistency.
"""

import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from controlnet_aux import CannyDetector
from PIL import Image
from typing import Optional, List
import numpy as np
import cv2


class ControlNetFrameGenerator:
    """
    Generates frames using ControlNet for better temporal consistency.
    Uses edge detection (Canny) to maintain structural coherence across frames.
    """
    
    def __init__(
        self,
        model_id: str = "Manojb/stable-diffusion-2-1-base",
        controlnet_id: str = "thibaud/controlnet-sd21-canny-diffusers",
        device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        torch_dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    ):
        """
        Initialize ControlNet pipeline.
        
        Args:
            model_id: Base Stable Diffusion model
            controlnet_id: ControlNet model for conditioning
            device: Device to run on
            torch_dtype: Data type for model weights
        """
        self.device = device
        self.torch_dtype = torch_dtype
        
        print(f"Loading ControlNet model: {controlnet_id}")
        print(f"Device: {device}")
        
        # Load ControlNet
        self.controlnet = ControlNetModel.from_pretrained(
            controlnet_id,
            torch_dtype=torch_dtype
        )
        
        # Load pipeline
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            model_id,
            controlnet=self.controlnet,
            torch_dtype=torch_dtype,
            safety_checker=None
        ).to(device)
        
        # Enable optimizations
        if device in ("cuda", "mps"):
            self.pipe.enable_attention_slicing()
        
        # Initialize Canny detector
        self.canny_detector = CannyDetector()
        
        print("ControlNet pipeline loaded successfully!")
    
    def extract_canny_edges(
        self,
        image: Image.Image,
        low_threshold: int = 100,
        high_threshold: int = 200
    ) -> Image.Image:
        """
        Extract Canny edge map from an image.
        
        Args:
            image: Input PIL Image
            low_threshold: Lower threshold for edge detection
            high_threshold: Upper threshold for edge detection
            
        Returns:
            PIL Image of edge map
        """
        # Use controlnet_aux detector
        canny_image = self.canny_detector(
            image,
            low_threshold=low_threshold,
            high_threshold=high_threshold
        )
        return canny_image
    
    def generate_frame_from_edges(
        self,
        edge_map: Image.Image,
        prompt: str,
        negative_prompt: str = "blurry, bad quality, distorted",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 1.0,
        seed: Optional[int] = None
    ) -> Image.Image:
        """
        Generate a frame conditioned on edge map.
        
        Args:
            edge_map: Canny edge map to condition on
            prompt: Text description
            negative_prompt: What to avoid
            num_inference_steps: Denoising steps
            guidance_scale: Prompt adherence
            controlnet_conditioning_scale: ControlNet influence strength
            seed: Random seed
            
        Returns:
            Generated PIL Image
        """
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)
        
        output = self.pipe(
            prompt=prompt,
            image=edge_map,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=generator
        )
        
        return output.images[0]
    
    def _warp_edges(
        self,
        edge_image: Image.Image,
        frame_idx: int,
        strength: float,
        max_shift: int = 8
    ) -> Image.Image:
        """
        Apply a smooth progressive warp to an edge map to simulate motion.

        Uses sinusoidal displacement so each frame shifts the edges slightly
        differently, creating continuous motion rather than a static loop.

        Args:
            edge_image: Canny edge map as PIL Image
            frame_idx: Current frame index (drives warp progression)
            strength: Warp strength (0.0-1.0)
            max_shift: Maximum pixel displacement at full strength

        Returns:
            Warped edge map as PIL Image
        """
        img_np = np.array(edge_image)
        h, w = img_np.shape[:2]

        # Sinusoidal progression keeps motion smooth across frames
        shift_x = strength * max_shift * np.sin(frame_idx * 0.3)
        shift_y = strength * max_shift * np.cos(frame_idx * 0.2)

        # Build remap grids with the displacement applied
        x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (x_coords + shift_x).astype(np.float32)
        map_y = (y_coords + shift_y).astype(np.float32)

        warped = cv2.remap(
            img_np, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

        return Image.fromarray(warped)

    def generate_consistent_sequence(
        self,
        initial_frame: Image.Image,
        prompt: str,
        num_frames: int = 16,
        guidance_scale: float = 7.5,
        controlnet_scale: float = 0.8,
        edge_variation: float = 0.0,
        seed: Optional[int] = None,
        progress_callback: Optional[callable] = None
    ) -> List[Image.Image]:
        """
        Generate temporally consistent frame sequence using ControlNet.
        
        Args:
            initial_frame: First frame to start from
            prompt: Text description
            num_frames: Number of frames to generate
            guidance_scale: Prompt adherence
            controlnet_scale: ControlNet influence (lower = more variation)
            edge_variation: Slight edge map variation for motion (0.0-1.0)
            seed: Random seed
            progress_callback: Progress update callback
            
        Returns:
            List of generated frames
        """
        frames = [initial_frame]
        
        for i in range(1, num_frames):
            if progress_callback:
                progress_callback(i, num_frames, f"Generating frame {i+1}/{num_frames}...")
            
            # Extract edges from previous frame
            edge_map = self.extract_canny_edges(frames[-1])
            
            # Apply progressive warp to edges to simulate motion
            if edge_variation > 0:
                edge_map = self._warp_edges(edge_map, frame_idx=i, strength=edge_variation)
            
            # Generate next frame conditioned on edges
            next_frame = self.generate_frame_from_edges(
                edge_map=edge_map,
                prompt=prompt,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_scale,
                seed=seed + i if seed else None
            )
            
            frames.append(next_frame)
        
        if progress_callback:
            progress_callback(num_frames, num_frames, "Complete!")
        
        return frames


if __name__ == "__main__":
    # Test ControlNet pipeline
    print("ControlNet pipeline module loaded.")
    print("Run with a base frame to test full functionality.")

