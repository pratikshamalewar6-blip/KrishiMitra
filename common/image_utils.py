"""
KrishiMitra - Image Utilities

Production-ready utility functions for image validation,
loading, preprocessing and conversion.

Features
--------
- Safe image loading
- Corruption detection
- EXIF orientation correction
- RGB conversion
- Supported format validation
- Basic image information

Author:
Pratiksha Malewar
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image
from PIL import ImageOps
from PIL import UnidentifiedImageError

from common.logger import LoggerManager


class ImageUtils:
    """
    Production utility class for image operations.

    All methods are static because no instance state
    is required.
    """

    logger = LoggerManager.get_logger("ImageUtils")

    SUPPORTED_FORMATS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    }

    # ==========================================================
    # Format Validation
    # ==========================================================

    @staticmethod
    def is_supported_image(path: str | Path) -> bool:
        """
        Check whether the given file has a supported image extension.

        Parameters
        ----------
        path : str | Path

        Returns
        -------
        bool
        """

        return Path(path).suffix.lower() in ImageUtils.SUPPORTED_FORMATS

    # ==========================================================
    # Image Verification
    # ==========================================================

    @staticmethod
    def verify_image(path: str | Path) -> bool:
        """
        Verify whether an image is valid.

        This method detects corrupted images without
        fully loading them into memory.

        Parameters
        ----------
        path : str | Path

        Returns
        -------
        bool
        """

        image_path = Path(path)

        if not image_path.exists():

            ImageUtils.logger.error(
                f"Image not found: {image_path}"
            )

            return False

        if not ImageUtils.is_supported_image(image_path):

            ImageUtils.logger.warning(
                f"Unsupported image format: {image_path.name}"
            )

            return False

        try:

            with Image.open(image_path) as image:

                image.verify()

            return True

        except (UnidentifiedImageError, OSError, ValueError) as error:

            ImageUtils.logger.error(
                f"Corrupted image detected: {image_path} | {error}"
            )

            return False

    # ==========================================================
    # Safe Image Loading
    # ==========================================================

    @staticmethod
    def load_image(path: str | Path) -> Optional[Image.Image]:
        """
        Safely load an image.

        Automatically verifies the image before loading.

        Returns
        -------
        PIL.Image.Image | None
        """

        image_path = Path(path)

        if not ImageUtils.verify_image(image_path):

            return None

        try:

            image = Image.open(image_path)

            image = ImageOps.exif_transpose(image)

            return image

        except Exception as error:

            ImageUtils.logger.error(
                f"Failed to load image: {image_path}\n{error}"
            )

            return None

    # ==========================================================
    # RGB Conversion
    # ==========================================================

    @staticmethod
    def convert_rgb(image: Image.Image) -> Image.Image:
        """
        Convert image to RGB.

        Handles:

        RGB
        RGBA
        Grayscale
        Palette

        Parameters
        ----------
        image : PIL.Image.Image

        Returns
        -------
        PIL.Image.Image
        """

        if image.mode == "RGB":

            return image

        ImageUtils.logger.info(
            f"Converting image mode "
            f"{image.mode} -> RGB"
        )

        return image.convert("RGB")

    # ==========================================================
    # EXIF Orientation
    # ==========================================================

    @staticmethod
    def fix_orientation(image: Image.Image) -> Image.Image:
        """
        Correct image orientation using EXIF metadata.

        Images captured from smartphones often contain
        EXIF orientation tags.

        Returns
        -------
        PIL.Image.Image
        """

        try:

            return ImageOps.exif_transpose(image)

        except Exception:

            return image

    # ==========================================================
    # Image Dimensions
    # ==========================================================

    @staticmethod
    def get_size(image: Image.Image) -> tuple[int, int]:
        """
        Get image width and height.

        Returns
        -------
        (width, height)
        """

        return image.size

    @staticmethod
    def get_width(image: Image.Image) -> int:
        """
        Get image width.
        """

        return image.width

    @staticmethod
    def get_height(image: Image.Image) -> int:
        """
        Get image height.
        """

        return image.height

    # ==========================================================
    # Image Mode
    # ==========================================================

    @staticmethod
    def get_mode(image: Image.Image) -> str:
        """
        Get image mode.

        Example
        -------
        RGB
        RGBA
        L
        """

        return image.mode

    # ==========================================================
    # Basic Information
    # ==========================================================

    @staticmethod
    def print_summary(image: Image.Image) -> None:
        """
        Print a basic summary of an image.

        Useful for debugging.
        """

        width, height = image.size

        ImageUtils.logger.info(
            "Image Summary\n"
            f"Width  : {width}\n"
            f"Height : {height}\n"
            f"Mode   : {image.mode}"
        )

# ==========================================================
# Image Metadata
# ==========================================================

    @staticmethod
    def get_metadata(path: str | Path) -> dict:
        """
        Extract image metadata.

        Parameters
        ----------
        path : str | Path

        Returns
        -------
        dict
        """

        image = ImageUtils.load_image(path)

        if image is None:
            return {}

        width, height = image.size

        channels = len(image.getbands())

        megapixels = round((width * height) / 1_000_000, 2)

        aspect_ratio = round(width / height, 3)

        return {
            "filename": Path(path).name,
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            "megapixels": megapixels,
            "channels": channels,
            "mode": image.mode,
            "format": image.format,
        }

# ==========================================================
# Image Hash
# ==========================================================

    @staticmethod
    def compute_sha256(path: str | Path) -> str:
        """
        Compute SHA256 hash of an image.

        Useful for duplicate detection and dataset integrity.
        """

        import hashlib

        sha = hashlib.sha256()

        with open(path, "rb") as file:

            while True:

                chunk = file.read(4096)

                if not chunk:
                    break

                sha.update(chunk)

        return sha.hexdigest()

# ==========================================================
# PIL <-> NumPy
# ==========================================================

    @staticmethod
    def pil_to_numpy(image: Image.Image):
        """
        Convert PIL image to NumPy array.
        """

        import numpy as np

        return np.array(image)

    @staticmethod
    def numpy_to_pil(array):
        """
        Convert NumPy array to PIL image.
        """

        return Image.fromarray(array)

# ==========================================================
# PIL <-> OpenCV
# ==========================================================

    @staticmethod
    def pil_to_cv2(image: Image.Image):
        """
        Convert PIL image to OpenCV image.
        """

        import cv2

        import numpy as np

        rgb = np.array(image)

        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    @staticmethod
    def cv2_to_pil(image):
        """
        Convert OpenCV image to PIL.
        """

        import cv2

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return Image.fromarray(rgb)

# ==========================================================
# Resize
# ==========================================================

    @staticmethod
    def resize(
        image: Image.Image,
        size: tuple[int, int],
    ) -> Image.Image:
        """
        Resize image.

        Parameters
        ----------
        image
        size

        Returns
        -------
        PIL.Image
        """

        return image.resize(size)

# ==========================================================
# Thumbnail
# ==========================================================

    @staticmethod
    def thumbnail(
        image: Image.Image,
        max_size: tuple[int, int] = (256, 256),
    ) -> Image.Image:
        """
        Create thumbnail.
        """

        img = image.copy()

        img.thumbnail(max_size)

        return img

# ==========================================================
# Save Image
# ==========================================================

    @staticmethod
    def save_image(
        image: Image.Image,
        output_path: str | Path,
    ) -> None:
        """
        Save image.
        """

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        image.save(output_path)

        ImageUtils.logger.info(
            f"Saved image: {output_path}"
        )
