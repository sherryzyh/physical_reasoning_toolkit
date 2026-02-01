"""
Utility functions for model clients.

This module provides common utility functions that can be used across different
model client implementations.
"""

import base64


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
