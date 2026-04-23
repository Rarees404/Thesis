import base64
from io import BytesIO
from typing import List, Tuple

from PIL import Image


def resize_images(
    images: List[Image.Image],
    image_size: List[int] | Tuple[int],
) -> List[Image.Image]:
    """Resize and ensure RGB mode for every image. Converts before resizing to avoid
    palette/RGBA artifacts. Handles empty lists gracefully."""
    return [
        image.convert("RGB").resize((image_size[0], image_size[1]), Image.Resampling.BICUBIC)
        for image in images
    ]

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def base64_to_image(base64_string: str) -> Image.Image:
    """Convert base64 string to PIL Image"""
    return Image.open(BytesIO(base64.b64decode(base64_string)))
