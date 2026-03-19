import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline


MODEL_ID = "stabilityai/stable-diffusion-2-inpainting"


class Inpainter:
    def __init__(self):
        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32

        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
        ).to(self.device)

        self.pipe.enable_attention_slicing()

    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str = "background",
        negative_prompt: str = "object, artifact, blurry, distorted",
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        seed: int | None = None,
    ) -> Image.Image:
        """
        Erase the masked region and fill it with generated background.

        Args:
            image:               Original RGB image
            mask:                Binary mask — 255 = erase, 0 = keep (mode "L")
            prompt:              What to fill the erased area with
            negative_prompt:     What to avoid generating
            guidance_scale:      Classifier-free guidance strength
            num_inference_steps: Diffusion steps
            seed:                Optional RNG seed for reproducibility

        Returns:
            PIL Image with the masked region inpainted
        """
        generator = (
            torch.Generator(device=self.device).manual_seed(seed)
            if seed is not None else None
        )

        # SD inpainting expects both image and mask at the same size
        w, h = image.size
        mask_resized = mask.resize((w, h), Image.NEAREST)

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image.convert("RGB"),
            mask_image=mask_resized,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images[0]

        return result
