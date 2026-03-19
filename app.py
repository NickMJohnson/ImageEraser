import numpy as np
import gradio as gr
from PIL import Image, ImageDraw

from src.segment import SegmentAnything
from src.inpaint import Inpainter


# ---------------------------------------------------------------------------
# Model loading (once at startup)
# ---------------------------------------------------------------------------

segmenter = SegmentAnything()
inpainter = Inpainter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def overlay_mask(image: Image.Image, mask: Image.Image, color=(255, 0, 0), alpha=0.5) -> Image.Image:
    """Blend a red mask overlay onto the original image for preview."""
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    mask_arr = np.array(mask)
    ys, xs = np.where(mask_arr > 128)
    for x, y in zip(xs, ys):
        draw.point((x, y), fill=(*color, int(255 * alpha)))
    return Image.alpha_composite(base, overlay).convert("RGB")


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------

def on_image_upload(image: Image.Image, state: dict) -> dict:
    """Encode the uploaded image into SAM so it's ready for point queries."""
    segmenter.set_image(image)
    state["image"] = image
    state["mask"] = None
    return state


def on_select(image: Image.Image, state: dict, evt: gr.SelectData) -> tuple:
    """
    Triggered when the user clicks a point on the image.
    Runs SAM and returns a mask-overlay preview.
    """
    if state.get("image") is None:
        return image, state

    x, y = evt.index  # pixel coordinates of the click
    mask = segmenter.predict_mask(
        point_coords=[(x, y)],
        point_labels=[1],
    )
    state["mask"] = mask
    preview = overlay_mask(state["image"], mask)
    return preview, state


def on_erase(
    state: dict,
    prompt: str,
    negative_prompt: str,
    guidance_scale: float,
    num_steps: int,
    seed: int,
) -> Image.Image:
    """Run SD inpainting on the selected mask."""
    if state.get("image") is None or state.get("mask") is None:
        raise gr.Error("Upload an image and click an object first.")

    result = inpainter.inpaint(
        image=state["image"],
        mask=state["mask"],
        prompt=prompt or "background",
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=int(num_steps),
        seed=seed if seed >= 0 else None,
    )
    return result


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Image Eraser") as demo:
    gr.Markdown("# Image Eraser\nUpload an image, click an object to select it, then hit **Erase**.")

    state = gr.State({})

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload image", type="pil", interactive=True)
            preview_image = gr.Image(label="Selection preview", interactive=True)

        with gr.Column():
            prompt = gr.Textbox(label="Fill prompt", value="background", lines=1)
            negative_prompt = gr.Textbox(
                label="Negative prompt",
                value="object, artifact, blurry, distorted",
                lines=1,
            )
            with gr.Row():
                guidance_scale = gr.Slider(1, 15, value=7.5, step=0.5, label="Guidance scale")
                num_steps = gr.Slider(10, 100, value=50, step=5, label="Steps")
            seed = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)
            erase_btn = gr.Button("Erase", variant="primary")
            output_image = gr.Image(label="Result")

    # Wire up events
    input_image.upload(on_image_upload, inputs=[input_image, state], outputs=[state])
    preview_image.select(on_select, inputs=[input_image, state], outputs=[preview_image, state])
    input_image.select(on_select, inputs=[input_image, state], outputs=[preview_image, state])
    erase_btn.click(
        on_erase,
        inputs=[state, prompt, negative_prompt, guidance_scale, num_steps, seed],
        outputs=[output_image],
    )


if __name__ == "__main__":
    demo.launch()
