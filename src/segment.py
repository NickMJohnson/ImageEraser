import numpy as np
from PIL import Image
import torch
from segment_anything import SamPredictor, sam_model_registry


class SegmentAnything:
    def __init__(self, checkpoint: str = "./checkpoints/sam_vit_h.pth", model_type: str = "vit_h"):
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device)
        self.predictor = SamPredictor(sam)

    def set_image(self, image: Image.Image) -> None:
        """Encode an image so it's ready for point-prompt queries."""
        rgb = np.array(image.convert("RGB"))
        self.predictor.set_image(rgb)

    def predict_mask(
        self,
        point_coords: list[tuple[int, int]],
        point_labels: list[int],
    ) -> Image.Image:
        """
        Run SAM for one or more point prompts and return the best binary mask.

        Args:
            point_coords: list of (x, y) pixel coordinates
            point_labels: 1 = foreground, 0 = background (one per coord)

        Returns:
            PIL Image in mode "L" — 255 = masked (object), 0 = keep
        """
        coords = np.array(point_coords, dtype=np.float32)
        labels = np.array(point_labels, dtype=np.int32)

        masks, scores, _ = self.predictor.predict(
            point_coords=coords,
            point_labels=labels,
            multimask_output=True,
        )

        # Pick the mask with the highest confidence score
        best = masks[scores.argmax()]
        return Image.fromarray((best * 255).astype(np.uint8), mode="L")
