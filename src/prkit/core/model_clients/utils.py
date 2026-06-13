"""
Utility functions for model clients.

This module provides common utility functions that can be used across different
model client implementations.
"""

import base64
import os


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode an image file to base64 data string format.

    Args:
        image_path: Path to the image file

    Returns:
        Base64-encoded data string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def detect_image_mime_type(image_path: str) -> str:
    """Detect an image MIME type from a file path extension.

    Args:
        image_path: Image file path or filename.

    Returns:
        MIME type string. Unknown extensions default to ``image/jpeg``.
    """
    ext = os.path.splitext(image_path)[1].lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return mime_types.get(ext, "image/jpeg")


def parse_data_url(data_url: str) -> dict[str, str]:
    """Parse a base64 data URL into media type and payload.

    Args:
        data_url: URL in ``data:<media_type>;base64,<payload>`` format.

    Returns:
        Dict with ``media_type`` and ``data`` keys.

    Raises:
        ValueError: If the URL is not a base64 data URL.
    """
    if not data_url.startswith("data:"):
        raise ValueError("Expected data URL to start with 'data:'")
    if ";base64," not in data_url:
        raise ValueError("Image data URL must include ';base64,'")

    header, payload = data_url.split(",", 1)
    media_type = header[5:].split(";")[0] or "image/jpeg"
    return {"media_type": media_type, "data": payload}


def prepare_image_url_from_path(image_path: str) -> str:
    """Prepare an image URL from a file path, HTTP URL, or data URL.

    Args:
        image_path: Local file path, HTTP(S) URL, or base64 data URL.

    Returns:
        Original URL/data URL, or a generated base64 data URL for local files.

    Raises:
        FileNotFoundError: If a local file path does not exist.
    """
    if image_path.startswith(("data:", "http://", "https://")):
        return image_path

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    mime_type = detect_image_mime_type(image_path)
    base64_image_string = encode_image_to_base64(image_path)
    return f"data:{mime_type};base64,{base64_image_string}"
