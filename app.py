import gradio as gr
import torch
from PIL import Image
import numpy as np
from model import FluxModel
import os

def load_image(image_path):
    """Load and return a PIL Image."""
    try:
        return Image.open(image_path).convert('RGB')
    except Exception as e:
        raise ValueError(f"Error loading image {image_path}: {str(e)}")

def save_images(images, output_dir, prefix="generated"):
    """Save generated images with sequential numbering."""
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []

    for i, image in enumerate(images):
        output_path = os.path.join(output_dir, f"{prefix}_{i+1}.png")
        image.save(output_path)
        saved_paths.append(output_path)

    return saved_paths

def get_required_features(mode, line_mode, depth_mode):
    """Determine which model features are required based on the arguments."""
    features = []
    if mode in ['controlnet', 'controlnet-inpaint']:
        features.append('controlnet')
        if depth_mode:
            features.append('depth')
        if line_mode:
            features.append('line')

    if mode in ['inpaint', 'controlnet-inpaint']:
        features.append('sam')  # If you're using SAM for mask generation

    return features

def generate_images(mode, input_image, prompt, reference_image, mask_image, output_dir,
                    image_count, aspect_ratio, steps, guidance_scale, denoise_strength,
                    center_x, center_y, radius, line_mode, depth_mode, line_strength,
                    depth_strength, turbo):

    # Check CUDA availability if requested
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Determine required features
    required_features = get_required_features(mode, line_mode, depth_mode)

    # Initialize model
    model = FluxModel(
        is_turbo=turbo,
        device=device,
        required_features=required_features
    )

    # Load input images
    input_image = load_image(input_image)
    reference_image = load_image(reference_image) if reference_image else None
    mask_image = load_image(mask_image) if mask_image else None

    # Validate inputs based on mode
    if mode in ['inpaint', 'controlnet-inpaint'] and mask_image is None:
        raise ValueError(f"{mode} mode requires a mask image")

    # Generate images
    generated_images = model.generate(
        input_image_a=input_image,
        input_image_b=reference_image,
        prompt=prompt,
        mask_image=mask_image,
        mode=mode,
        imageCount=image_count,
        aspect_ratio=aspect_ratio,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        denoise_strength=denoise_strength,
        center_x=center_x,
        center_y=center_y,
        radius=radius,
        line_mode=line_mode,
        depth_mode=depth_mode,
        line_strength=line_strength,
        depth_strength=depth_strength
    )

    # Save generated images
    return save_images(generated_images, output_dir)

# Gradio interface
def gradio_app():
    with gr.Blocks() as app:
        gr.Markdown("# Flux Image Generation Tool")

        # Input components
        mode = gr.Radio(["variation", "img2img", "inpaint", "controlnet", "controlnet-inpaint"], label="Mode", value="variation")
        input_image = gr.Image(type="filepath", label="Input Image", tool="editor")
        prompt = gr.Textbox(label="Text Prompt", value="")
        reference_image = gr.Image(type="filepath", label="Reference Image")
        mask_image = gr.Image(type="filepath", label="Mask Image")
        output_dir = gr.Textbox(label="Output Directory", value="outputs")
        image_count = gr.Slider(1, 10, step=1, value=1, label="Image Count")
        aspect_ratio = gr.Dropdown(["1:1", "16:9", "9:16", "2.4:1", "3:4", "4:3"], label="Aspect Ratio", value="1:1")
        steps = gr.Slider(1, 100, step=1, value=28, label="Steps")
        guidance_scale = gr.Slider(0.0, 20.0, step=0.1, value=7.5, label="Guidance Scale")
        denoise_strength = gr.Slider(0.0, 1.0, step=0.1, value=0.8, label="Denoise Strength")
        center_x = gr.Slider(0.0, 1.0, step=0.1, label="Center X")
        center_y = gr.Slider(0.0, 1.0, step=0.1, label="Center Y")
        radius = gr.Slider(0.0, 1.0, step=0.1, value=None, label="Radius", visible=False)
        line_mode = gr.Checkbox(label="Line Mode")
        depth_mode = gr.Checkbox(label="Depth Mode")
        line_strength = gr.Slider(0.0, 1.0, step=0.1, value=0.4, label="Line Strength")
        depth_strength = gr.Slider(0.0, 1.0, step=0.1, value=0.2, label="Depth Strength")
        turbo = gr.Checkbox(label="Turbo Mode")

        # Output component
        output_gallery = gr.Gallery(label="Generated Images")

        # Submit button
        generate_button = gr.Button("Generate Images")

        # Link button to function
        generate_button.click(
            generate_images,
            inputs=[mode, input_image, prompt, reference_image, mask_image, output_dir, image_count, aspect_ratio,
                    steps, guidance_scale, denoise_strength, center_x, center_y, radius, line_mode, depth_mode,
                    line_strength, depth_strength, turbo],
            outputs=output_gallery
        )

    return app

if __name__ == "__main__":
    app = gradio_app()
    app.launch()
