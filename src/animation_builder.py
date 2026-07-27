"""
Animation Builder Module
Converts frame sequences into GIF/MP4 animations.
"""

from PIL import Image
from typing import List, Optional
import imageio
import os
from pathlib import Path


class AnimationBuilder:
    """
    Builds animations (GIF/MP4) from frame sequences.
    """
    
    def __init__(self, output_dir: str = "outputs"):
        """
        Initialize animation builder.
        
        Args:
            output_dir: Directory to save animations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {self.output_dir.absolute()}")
    
    def save_frames(
        self,
        frames: List[Image.Image],
        prefix: str = "frame",
        subfolder: Optional[str] = None
    ) -> List[Path]:
        """
        Save individual frames as PNG files.
        
        Args:
            frames: List of PIL Images
            prefix: Filename prefix
            subfolder: Optional subfolder name
            
        Returns:
            List of saved file paths
        """
        if subfolder:
            save_dir = self.output_dir / subfolder
        else:
            save_dir = self.output_dir
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        for i, frame in enumerate(frames):
            filename = f"{prefix}_{i:04d}.png"
            filepath = save_dir / filename
            frame.save(filepath)
            saved_paths.append(filepath)
        
        print(f"Saved {len(frames)} frames to {save_dir}")
        return saved_paths
    
    def create_gif(
        self,
        frames: List[Image.Image],
        output_filename: str = "animation.gif",
        fps: int = 8,
        loop: int = 0,
        optimize: bool = True
    ) -> Path:
        """
        Create GIF animation from frames.
        
        Args:
            frames: List of PIL Images
            output_filename: Output GIF filename
            fps: Frames per second
            loop: Number of loops (0 = infinite)
            optimize: Optimize GIF size
            
        Returns:
            Path to saved GIF
        """
        output_path = self.output_dir / output_filename
        
        # Calculate duration per frame in milliseconds
        duration_ms = int(1000 / fps)
        
        # Save as GIF
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=loop,
            optimize=optimize
        )
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"GIF saved: {output_path} ({file_size_mb:.2f} MB)")
        
        return output_path
    
    def create_mp4(
        self,
        frames: List[Image.Image],
        output_filename: str = "animation.mp4",
        fps: int = 8,
        quality: int = 8
    ) -> Path:
        """
        Create MP4 video from frames.
        
        Args:
            frames: List of PIL Images
            output_filename: Output MP4 filename
            fps: Frames per second
            quality: Video quality (1-10, 10 is best)
            
        Returns:
            Path to saved MP4
        """
        output_path = self.output_dir / output_filename
        
        # Convert PIL Images to numpy arrays
        frame_arrays = [imageio.core.util.Array(frame) for frame in frames]
        
        # Write video
        imageio.mimsave(
            output_path,
            frame_arrays,
            fps=fps,
            quality=quality,
            codec='libx264',
            pixelformat='yuv420p'
        )
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"MP4 saved: {output_path} ({file_size_mb:.2f} MB)")
        
        return output_path
    
    def create_side_by_side_comparison(
        self,
        frames_a: List[Image.Image],
        frames_b: List[Image.Image],
        output_filename: str = "comparison.gif",
        fps: int = 8,
        labels: Optional[tuple] = None
    ) -> Path:
        """
        Create side-by-side comparison animation.
        
        Args:
            frames_a: First frame sequence
            frames_b: Second frame sequence
            output_filename: Output filename
            fps: Frames per second
            labels: Optional tuple of labels for each sequence
            
        Returns:
            Path to saved animation
        """
        combined_frames = []
        
        for frame_a, frame_b in zip(frames_a, frames_b):
            # Ensure same height
            height = max(frame_a.height, frame_b.height)
            frame_a = frame_a.resize((int(frame_a.width * height / frame_a.height), height))
            frame_b = frame_b.resize((int(frame_b.width * height / frame_b.height), height))
            
            # Combine horizontally
            combined = Image.new('RGB', (frame_a.width + frame_b.width, height))
            combined.paste(frame_a, (0, 0))
            combined.paste(frame_b, (frame_a.width, 0))
            
            combined_frames.append(combined)
        
        return self.create_gif(combined_frames, output_filename, fps)
    
    def create_grid_animation(
        self,
        frame_sequences: List[List[Image.Image]],
        output_filename: str = "grid.gif",
        fps: int = 8,
        grid_size: Optional[tuple] = None
    ) -> Path:
        """
        Create grid animation showing multiple sequences.
        
        Args:
            frame_sequences: List of frame sequences
            output_filename: Output filename
            fps: Frames per second
            grid_size: Optional (rows, cols) for grid layout
            
        Returns:
            Path to saved animation
        """
        num_sequences = len(frame_sequences)
        
        if grid_size is None:
            # Auto-calculate grid size
            cols = int(num_sequences ** 0.5)
            rows = (num_sequences + cols - 1) // cols
        else:
            rows, cols = grid_size
        
        # Get frame count (use minimum)
        num_frames = min(len(seq) for seq in frame_sequences)
        
        grid_frames = []
        
        for frame_idx in range(num_frames):
            # Get all frames at this index
            current_frames = [seq[frame_idx] for seq in frame_sequences]
            
            # Resize all to same size
            target_size = (256, 256)
            current_frames = [f.resize(target_size) for f in current_frames]
            
            # Create grid
            grid_width = cols * target_size[0]
            grid_height = rows * target_size[1]
            grid_image = Image.new('RGB', (grid_width, grid_height))
            
            for i, frame in enumerate(current_frames):
                row = i // cols
                col = i % cols
                x = col * target_size[0]
                y = row * target_size[1]
                grid_image.paste(frame, (x, y))
            
            grid_frames.append(grid_image)
        
        return self.create_gif(grid_frames, output_filename, fps)


if __name__ == "__main__":
    print("Animation builder module loaded.")
    print("Use to convert frame sequences into GIF/MP4 animations.")

