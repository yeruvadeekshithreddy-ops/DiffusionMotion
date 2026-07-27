"""
Gradio Web Interface for DiffusionMotion
AI-Assisted Animation Generator
"""

try:
    import spaces  # ZeroGPU support on HF Spaces — must be imported before torch
    HF_SPACES = True
except ImportError:
    spaces = None
    HF_SPACES = False

import gradio as gr
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.frame_generator import FrameGenerator
from src.controlnet_pipeline import ControlNetFrameGenerator
from src.interpolation import LatentInterpolator
from src.animation_builder import AnimationBuilder

MODE_BASIC = "Basic Img2Img"
MODE_CONTROLNET = "ControlNet"
MODE_INTERPOLATION = "Latent Interpolation"


class AnimationApp:
    """
    Main application class for the Gradio interface.
    Supports three generation modes: Basic, ControlNet, and Latent Interpolation.
    """

    def __init__(self):
        self.frame_generator = None
        self.controlnet_generator = None
        self.interpolator = None
        self.animation_builder = AnimationBuilder(output_dir="outputs")
        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

    # ------------------------------------------------------------------
    # Lazy model loaders
    # ------------------------------------------------------------------

    def _load_frame_generator(self):
        if self.frame_generator is None:
            print("Loading Stable Diffusion (txt2img / img2img)...")
            self.frame_generator = FrameGenerator(device=self.device)
        return self.frame_generator

    def _load_controlnet(self):
        if self.controlnet_generator is None:
            print("Loading ControlNet pipeline...")
            self.controlnet_generator = ControlNetFrameGenerator(device=self.device)
        return self.controlnet_generator

    def _load_interpolator(self):
        if self.interpolator is None:
            print("Loading VAE for latent interpolation...")
            self.interpolator = LatentInterpolator(device=self.device)
        return self.interpolator

    # ------------------------------------------------------------------
    # Generation entry point
    # ------------------------------------------------------------------

    def generate_animation(
        self,
        prompt: str,
        mode: str = MODE_BASIC,
        num_frames: int = 16,
        guidance_scale: float = 7.5,
        fps: int = 8,
        seed: int = -1,
        # Basic mode
        strength: float = 0.5,
        # ControlNet mode
        controlnet_scale: float = 0.8,
        edge_variation: float = 0.3,
        # Interpolation mode
        num_keyframes: int = 3,
        frames_between: int = 4,
        interpolation_type: str = "slerp",
        progress=gr.Progress()
    ):
        try:
            if not prompt or not prompt.strip():
                return None, "Please enter a prompt."
            if num_frames < 2:
                return None, "Need at least 2 frames."

            actual_seed = seed if seed >= 0 else torch.randint(0, 2**32, (1,)).item()

            def cb(current, total, msg):
                progress(current / total, desc=msg)

            # ── Basic Img2Img ─────────────────────────────────────────
            if mode == MODE_BASIC:
                progress(0, desc="Loading Stable Diffusion...")
                gen = self._load_frame_generator()
                progress(0.1, desc="Generating frames...")
                frames = gen.generate_frame_sequence(
                    base_prompt=prompt,
                    num_frames=num_frames,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    seed=actual_seed,
                    progress_callback=cb
                )
                mode_detail = f"Strength: {strength}"

            # ── ControlNet ────────────────────────────────────────────
            elif mode == MODE_CONTROLNET:
                progress(0, desc="Loading Stable Diffusion + ControlNet...")
                base_gen = self._load_frame_generator()
                cn_gen = self._load_controlnet()

                progress(0.05, desc="Generating initial frame...")
                initial_frame = base_gen.generate_first_frame(
                    prompt=prompt,
                    guidance_scale=guidance_scale,
                    seed=actual_seed
                )

                progress(0.1, desc="Generating ControlNet sequence...")
                frames = cn_gen.generate_consistent_sequence(
                    initial_frame=initial_frame,
                    prompt=prompt,
                    num_frames=num_frames,
                    guidance_scale=guidance_scale,
                    controlnet_scale=controlnet_scale,
                    edge_variation=edge_variation,
                    seed=actual_seed,
                    progress_callback=cb
                )
                mode_detail = f"ControlNet scale: {controlnet_scale} | Edge variation: {edge_variation}"

            # ── Latent Interpolation ──────────────────────────────────
            elif mode == MODE_INTERPOLATION:
                progress(0, desc="Loading Stable Diffusion + VAE...")
                base_gen = self._load_frame_generator()
                interp = self._load_interpolator()

                progress(0.05, desc=f"Generating {num_keyframes} keyframes...")
                keyframes = []
                for k in range(num_keyframes):
                    cb(k, num_keyframes, f"Keyframe {k + 1}/{num_keyframes}...")
                    kf = base_gen.generate_first_frame(
                        prompt=prompt,
                        guidance_scale=guidance_scale,
                        seed=actual_seed + k * 100
                    )
                    keyframes.append(kf)

                progress(0.6, desc="Interpolating in latent space...")
                frames = interp.generate_interpolated_sequence(
                    keyframes=keyframes,
                    frames_between=frames_between,
                    interpolation_type=interpolation_type
                )
                mode_detail = (
                    f"Keyframes: {num_keyframes} | "
                    f"Frames between: {frames_between} | "
                    f"Method: {interpolation_type}"
                )

            else:
                return None, f"Unknown mode: {mode}"

            # ── Export ────────────────────────────────────────────────
            progress(0.95, desc="Exporting GIF...")
            output_filename = f"animation_{mode.replace(' ', '_')}_{actual_seed}.gif"
            gif_path = self.animation_builder.create_gif(
                frames=frames,
                output_filename=output_filename,
                fps=fps
            )

            status_msg = f"""
**Animation Generated!**

**Mode:** {mode}
**Frames:** {len(frames)} @ {fps} FPS
**Seed:** {actual_seed}
**Device:** {self.device.upper()}
**Settings:** {mode_detail}

Saved to: `{gif_path}`
"""
            return str(gif_path), status_msg

        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, f"**Error:** {str(e)}\n\nCheck the console for details."


# Wrap generate_animation with @spaces.GPU on HF ZeroGPU, no-op locally
if HF_SPACES:
    AnimationApp.generate_animation = spaces.GPU(duration=300)(
        AnimationApp.generate_animation
    )


# ------------------------------------------------------------------
# UI helpers
# ------------------------------------------------------------------

def _toggle_mode_panels(mode):
    """Show/hide mode-specific setting panels based on selected mode."""
    return (
        gr.update(visible=(mode == MODE_BASIC)),
        gr.update(visible=(mode == MODE_CONTROLNET)),
        gr.update(visible=(mode == MODE_INTERPOLATION)),
    )


# ------------------------------------------------------------------
# Interface builder
# ------------------------------------------------------------------

def create_interface():
    app = AnimationApp()

    css = """
    .output-image { max-height: 500px; }
    .mode-panel { border: 1px solid #e0e0e0; border-radius: 8px; padding: 8px; }
    """

    with gr.Blocks(css=css, title="DiffusionMotion - AI Animation") as demo:

        gr.Markdown(
            """
            # DiffusionMotion — AI-Assisted Animation

            Generate smooth animations from text prompts using Stable Diffusion.

            | Mode | How it works | Best for |
            |------|-------------|----------|
            | **Basic Img2Img** | Chains img2img from the previous frame | Fast, general use |
            | **ControlNet** | Canny-edge conditioning for structural consistency | Objects that should hold their shape |
            | **Latent Interpolation** | SLERP/linear between VAE keyframes | Smooth morphs, abstract motion |
            """
        )

        with gr.Row():

            # ── Left column: inputs ───────────────────────────────────
            with gr.Column(scale=1):

                prompt_input = gr.Textbox(
                    label="Animation Prompt",
                    placeholder="A red ball bouncing up and down on a white background",
                    lines=3
                )

                mode_radio = gr.Radio(
                    choices=[MODE_BASIC, MODE_CONTROLNET, MODE_INTERPOLATION],
                    value=MODE_BASIC,
                    label="Generation Mode"
                )

                # ── Common settings ───────────────────────────────────
                with gr.Accordion("Common Settings", open=True):
                    num_frames_slider = gr.Slider(
                        minimum=2, maximum=32, value=16, step=1,
                        label="Number of Frames",
                        info="More frames = longer animation, slower generation"
                    )
                    guidance_slider = gr.Slider(
                        minimum=1.0, maximum=20.0, value=7.5, step=0.5,
                        label="Guidance Scale",
                        info="How closely to follow the prompt"
                    )
                    fps_slider = gr.Slider(
                        minimum=4, maximum=30, value=8, step=1,
                        label="FPS",
                        info="Playback speed"
                    )
                    seed_input = gr.Number(
                        label="Seed", value=-1, precision=0,
                        info="-1 for random"
                    )

                # ── Basic mode settings ───────────────────────────────
                with gr.Group(visible=True, elem_classes=["mode-panel"]) as basic_panel:
                    gr.Markdown("**Basic Img2Img Settings**")
                    strength_slider = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.5, step=0.05,
                        label="Frame Variation Strength",
                        info="Higher = more change between frames"
                    )

                # ── ControlNet mode settings ──────────────────────────
                with gr.Group(visible=False, elem_classes=["mode-panel"]) as controlnet_panel:
                    gr.Markdown("**ControlNet Settings**")
                    controlnet_scale_slider = gr.Slider(
                        minimum=0.1, maximum=1.5, value=0.8, step=0.05,
                        label="ControlNet Conditioning Scale",
                        info="Higher = stronger structural guidance"
                    )
                    edge_variation_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.3, step=0.05,
                        label="Edge Variation (Motion Amount)",
                        info="0 = static structure, higher = more edge movement"
                    )

                # ── Interpolation mode settings ───────────────────────
                with gr.Group(visible=False, elem_classes=["mode-panel"]) as interp_panel:
                    gr.Markdown("**Latent Interpolation Settings**")
                    num_keyframes_slider = gr.Slider(
                        minimum=2, maximum=8, value=3, step=1,
                        label="Number of Keyframes",
                        info="Keyframes generated at different seeds, then interpolated"
                    )
                    frames_between_slider = gr.Slider(
                        minimum=2, maximum=16, value=4, step=1,
                        label="Frames Between Keyframes",
                        info="Interpolated frames inserted between each keyframe pair"
                    )
                    interp_type_radio = gr.Radio(
                        choices=["slerp", "linear"],
                        value="slerp",
                        label="Interpolation Method",
                        info="SLERP is smoother for latent spaces"
                    )

                generate_btn = gr.Button(
                    "Generate Animation", variant="primary", size="lg"
                )

            # ── Right column: output ──────────────────────────────────
            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="Generated Animation",
                    type="filepath",
                    elem_classes=["output-image"]
                )
                status_output = gr.Markdown(
                    value="Ready. Enter a prompt and click Generate."
                )

        # ── Examples ──────────────────────────────────────────────────
        gr.Examples(
            examples=[
                ["A red ball bouncing up and down", MODE_BASIC,       16, 7.5, 8,  42,  0.4, 0.8, 0.3, 3, 4, "slerp"],
                ["A blue cube slowly rotating",     MODE_BASIC,       16, 7.5, 8,  123, 0.5, 0.8, 0.3, 3, 4, "slerp"],
                ["A fire flickering in the dark",   MODE_CONTROLNET,  16, 8.0, 8,  42,  0.5, 0.7, 0.4, 3, 4, "slerp"],
                ["A glowing orb pulsing",           MODE_CONTROLNET,  12, 7.5, 10, 99,  0.5, 0.8, 0.2, 3, 4, "slerp"],
                ["Abstract shapes morphing",        MODE_INTERPOLATION, 16, 7.5, 8, 42,  0.5, 0.8, 0.3, 3, 6, "slerp"],
                ["A face slowly smiling",           MODE_INTERPOLATION, 16, 8.0, 8, 200, 0.5, 0.8, 0.3, 4, 4, "slerp"],
            ],
            inputs=[
                prompt_input, mode_radio,
                num_frames_slider, guidance_slider, fps_slider, seed_input,
                strength_slider,
                controlnet_scale_slider, edge_variation_slider,
                num_keyframes_slider, frames_between_slider, interp_type_radio,
            ],
            label="Example Prompts"
        )

        # ── Footer ─────────────────────────────────────────────────────
        gr.Markdown(
            """
            ---
            **System Info:** Device: {} | Model: Stable Diffusion 2.1 Base

            First generation is slower (model download + load). Subsequent runs are fast.
            """.format(
                "CUDA GPU" if torch.cuda.is_available()
                else "Apple MPS" if torch.backends.mps.is_available()
                else "CPU"
            )
        )

        # ── Event wiring ───────────────────────────────────────────────
        mode_radio.change(
            fn=_toggle_mode_panels,
            inputs=[mode_radio],
            outputs=[basic_panel, controlnet_panel, interp_panel]
        )

        generate_btn.click(
            fn=app.generate_animation,
            inputs=[
                prompt_input, mode_radio,
                num_frames_slider, guidance_slider, fps_slider, seed_input,
                strength_slider,
                controlnet_scale_slider, edge_variation_slider,
                num_keyframes_slider, frames_between_slider, interp_type_radio,
            ],
            outputs=[output_image, status_output]
        )

    return demo


if __name__ == "__main__":
    print("=" * 60)
    print("DiffusionMotion - AI-Assisted Animation")
    print("=" * 60)
    device_label = (
        "CUDA GPU" if torch.cuda.is_available()
        else "Apple MPS" if torch.backends.mps.is_available()
        else "CPU"
    )
    print(f"Device: {device_label}")
    print("Starting Gradio interface...")
    print("=" * 60)

    demo = create_interface()
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        ssr_mode=False
    )
