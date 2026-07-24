"""
KrishiMitra
SAM2 Leaf Segmenter

Loads SAM2 model and segments leaf regions from background.

Author:
    Antigravity AI
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union
import numpy as np
from PIL import Image

from common.logger import LoggerManager
from common.file_utils import FileUtils
from segmentation.config import SegmentationConfig
from segmentation.model import SAM2ModelLoader


class LeafSegmenter:
    """
    SAM2 Leaf Segmenter.
    """

    def __init__(self, config: SegmentationConfig | None = None) -> None:
        self.logger = LoggerManager.get_logger("LeafSegmenter")
        self.config = config or SegmentationConfig()
        
        # Ensure output directory exists
        FileUtils.ensure_directory(self.config.OUTPUT_DIRECTORY)
        
        # Load the SAM2 model weights
        self.model = SAM2ModelLoader.load_model(self.config)

    def segment_leaf(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        bbox: Union[List[int], Tuple[int, int, int, int], 'DetectionResult'],
    ) -> Image.Image:
        """
        Segment a single leaf region out of an image using bounding box prompts.
        
        Parameters
        ----------
        image : str | Path | PIL.Image.Image | np.ndarray
            The source image containing the leaf.
        bbox : List[int] | Tuple[int, int, int, int] | DetectionResult
            Bounding box coordinates [x1, y1, x2, y2].

        Returns
        -------
        PIL.Image.Image
            Cropped leaf image with background removed.
        """
        # Convert image to NumPy RGB array
        if isinstance(image, (str, Path)):
            img_path = Path(image)
            if not img_path.exists():
                raise FileNotFoundError(f"Image file not found: {img_path}")
            pil_img = Image.open(img_path).convert("RGB")
            img_np = np.array(pil_img)
        elif isinstance(image, Image.Image):
            img_np = np.array(image.convert("RGB"))
        elif isinstance(image, np.ndarray):
            img_np = image.copy()
        else:
            raise TypeError("Unsupported image type. Must be file path, PIL Image, or NumPy array.")

        # Extract coordinates
        if hasattr(bbox, "x1"):  # Matches DetectionResult format
            x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
        else:
            x1, y1, x2, y2 = bbox

        H, W = img_np.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(W, int(x2)), min(H, int(y2))

        if x1 >= x2 or y1 >= y2:
            self.logger.warning(f"Invalid bounding box: [{x1}, {y1}, {x2}, {y2}]. Returning raw crop.")
            return Image.fromarray(img_np[y1:y2, x1:x2])

        # Predict segmentation mask with box prompts
        results = self.model.predict(
            source=img_np,
            bboxes=[[x1, y1, x2, y2]],
            device=self.config.DEVICE,
            verbose=False,
        )

        result = results[0]
        if result.masks is None or len(result.masks.data) == 0:
            self.logger.warning("SAM2 did not output any segmentation masks. Returning raw crop.")
            return Image.fromarray(img_np[y1:y2, x1:x2])

        # Extract the binary mask
        # Shape: (H, W)
        mask = result.masks.data[0].cpu().numpy().astype(bool)

        # Apply mask to background
        if self.config.OUTPUT_FORMAT.lower() == "png":
            # Premium transparent background output (RGBA)
            rgba_img = np.zeros((H, W, 4), dtype=np.uint8)
            rgba_img[:, :, :3] = img_np
            # Set alpha channel to 255 inside mask and 0 outside
            rgba_img[:, :, 3] = mask.astype(np.uint8) * 255
            
            # Crop to bounding box
            cropped_rgba = rgba_img[y1:y2, x1:x2]
            output_image = Image.fromarray(cropped_rgba, "RGBA")
        else:
            # Standard black background output (RGB)
            rgb_img = img_np * mask[:, :, np.newaxis]
            
            # Crop to bounding box
            cropped_rgb = rgb_img[y1:y2, x1:x2]
            output_image = Image.fromarray(cropped_rgb, "RGB")

        return output_image
