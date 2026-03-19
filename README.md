# Image Eraser

Remove objects from images by clicking on them. Uses Meta's Segment Anything Model (SAM) to select objects and Stable Diffusion inpainting to fill the erased region with realistic background.

![demo](assets/demo.gif)

## How it works

1. Upload an image
2. Click on the object you want to remove — SAM generates a mask overlay
3. Hit **Erase** — Stable Diffusion fills the region with background

## Stack

| Component | Library |
|---|---|
| Object selection | [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) |
| Inpainting | [Stable Diffusion 2 Inpainting](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting) via `diffusers` |
| Web UI | [Gradio](https://gradio.app) |

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the SAM checkpoint

```bash
mkdir -p checkpoints
curl -L -o checkpoints/sam_vit_h.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

### 3. Run

```bash
python app.py
```

The SD inpainting model (~5GB) will download from HuggingFace automatically on first run.

## Requirements

- Python 3.10+
- GPU strongly recommended (CUDA or Apple MPS) — CPU inference is slow
- ~8GB VRAM for SD inpainting at full quality

## Project Structure

```
image-eraser/
├── app.py              # Gradio UI
├── requirements.txt
├── src/
│   ├── segment.py      # SAM wrapper — click-to-mask
│   └── inpaint.py      # Stable Diffusion inpainting wrapper
└── checkpoints/        # SAM weights (not committed)
```
