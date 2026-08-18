"""Shared utility helpers for model client implementations (image encoding, MIME detection).

MIME detection prefers the file's magic bytes and falls back to its extension.
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


def sniff_image_media_type(image_path: str) -> str | None:
    """Detect an image MIME type from the file's magic bytes.

    Args:
        image_path: Image file path. The file is opened and its first 12 bytes read.

    Returns:
        MIME type string when the signature is recognized, otherwise ``None`` --
        including when the file cannot be read at all, so callers can fall back to
        the extension.
    """
    try:
        with open(image_path, "rb") as handle:
            header = handle.read(12)
    except OSError:
        return None

    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "image/webp"
    return None


def detect_image_mime_type(image_path: str) -> str:
    """Detect an image MIME type, preferring the file's bytes over its extension.

    Some dataset images carry a mismatched extension (PNG or GIF bytes in a ``.jpg``
    file). Providers validate the declared media type against the actual bytes and
    reject the request when the two disagree, so the file signature wins when it is
    recognizable.

    Args:
        image_path: Image file path or filename. A filename that names no readable
            file is still accepted; only the extension is consulted then.

    Returns:
        MIME type string, resolved from the magic bytes when they are recognized,
        then from the extension, defaulting to ``image/jpeg``.
    """
    sniffed = sniff_image_media_type(image_path)
    if sniffed is not None:
        return sniffed

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
