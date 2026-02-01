"""
OpenAI API client implementations.

This module provides client implementations for OpenAI's Responses API.

Supported OpenAI models:
- gpt-4.1 (and variants like gpt-4.1-mini, gpt-4.1-nano)
- gpt-5xxxx (gpt-5, gpt-5.1, gpt-5.2, gpt-5.1-mini, etc.)
- o-family (o3, o4, o4-mini, etc. - models starting with 'o' followed by number)
"""

import os
from typing import List, Optional

from openai import OpenAI

from .base import BaseModelClient
from .utils import encode_image_to_base64

def _is_supported_openai_model(model: str) -> bool:
    """
    Check if the OpenAI model is supported.
    
    Supported OpenAI models:
    - gpt-4.1 (and variants like gpt-4.1-mini, gpt-4.1-nano)
    - gpt-5xxxx (gpt-5, gpt-5.1, gpt-5.2, gpt-5.1-mini, etc.)
    - o-family (o3, o4, o4-mini, etc. - models starting with 'o' followed by number)
    
    Args:
        model: Model name to check
        
    Returns:
        True if the model is supported, False otherwise
    """
    model_lower = model.lower()
    
    # Check for o-family (o3, o4, o4-mini, etc. - starts with 'o' followed by number)
    if len(model_lower) > 1 and model_lower[0] == "o" and model_lower[1].isdigit():
        return True
    
    # Check for gpt-4.1
    if model_lower.startswith("gpt-4.1"):
        return True
    
    # Check for gpt-5xxxx
    if model_lower.startswith("gpt-5"):
        return True
    
    return False


def _is_o_family_model(model: str) -> bool:
    """
    Check if the model is an o-family reasoning model.
    
    Args:
        model: Model name to check
        
    Returns:
        True if the model is an o-family model, False otherwise
    """
    model_lower = model.lower()
    return len(model_lower) > 1 and model_lower[0] == "o" and model_lower[1].isdigit()



def prepare_image_url_from_image_path(image_path: str) -> str:
    """
    Prepare an image URL from a file path, URL, or base64 data URL.
    
    Args:
        image_path: Can be:
                   - File path: "/path/to/image.jpg" - will be encoded to base64
                   - HTTP/HTTPS URL: "https://example.com/image.jpg" - used as-is
                   - Base64 data URL: "data:image/jpeg;base64,..." - used as-is
        
    Returns:
        Image URL in the appropriate format:
        - Base64 data URL for file paths (e.g., "data:image/jpeg;base64,...")
        - Original URL for HTTP/HTTPS URLs
        - Original string for base64 data URLs
        
    Raises:
        FileNotFoundError: If image_path is a file path that doesn't exist
        IOError: If there's an error reading the image file
    """
    # If it's already a data URL, return as-is
    if image_path.startswith("data:"):
        return image_path
    
    # If it's an HTTP/HTTPS URL, return as-is
    if image_path.startswith("http://") or image_path.startswith("https://"):
        return image_path
    
    # Otherwise, treat it as a file path
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Determine MIME type from file extension
    ext = os.path.splitext(image_path)[1].lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
    }
    mime_type = mime_types.get(ext, 'image/jpeg')  # Default to jpeg
    
    base64_image_string = encode_image_to_base64(image_path)
    image_url = f"data:{mime_type};base64,{base64_image_string}"
    
    return image_url


class OpenAIModel(BaseModelClient):
    """OpenAI model client implementation using Responses API."""

    def __init__(self, model: str, logger=None):
        """
        Initialize OpenAI model client.

        Args:
            model: OpenAI model name. Supported models:
                  - gpt-4.1 (and variants like gpt-4.1-mini, gpt-4.1-nano)
                  - gpt-5xxxx (gpt-5, gpt-5.1, gpt-5.2, gpt-5.1-mini, etc.)
                  - o-family (o3, o4, o4-mini, etc. - models starting with 'o' followed by number)
            logger: Optional logger instance

        Raises:
            ValueError: If the model is not supported
        """
        if not _is_supported_openai_model(model):
            raise ValueError(
                f"Unsupported OpenAI model: {model}. "
                "Supported models: gpt-4.1 (and variants), gpt-5xxxx (gpt-5.1, gpt-5.2, etc.), "
                "and o-family (o3, o4, o4-mini, etc.)"
            )
        super().__init__(model, logger)
        self.client = OpenAI()
        self.provider = "openai"
        self.is_o_family = _is_o_family_model(model)

    def chat(self, user_prompt: str, image_paths: Optional[List[str]] = None):
        """
        Generate a response from OpenAI Responses API.

        Args:
            user_prompt: The user's prompt text (string)
            image_paths: Optional list of image paths/URLs (strings). Can be:
                       - File paths: ["/path/to/image.jpg", ...] - will be encoded to base64
                       - HTTP/HTTPS URLs: ["https://example.com/image.jpg", ...] - used as-is
                       - Base64 data URLs: ["data:image/jpeg;base64,...", ...] - used as-is

        Returns:
            Response text from OpenAI model
            
        Raises:
            FileNotFoundError: If any image_path is a file path that doesn't exist
            IOError: If there's an error reading any image file
        """
        # Build request parameters
        request_params = {"model": self.model}
        
        # Use role/content format for all models
        content = [{"type": "input_text", "text": user_prompt}]
        
        if image_paths:
            for image_path in image_paths:
                image_url = prepare_image_url_from_image_path(image_path)
                content.append({
                    "type": "input_image",
                    "image_url": image_url
                })
        
        request_params["input"] = [
            {
                "role": "user",
                "content": content
            }
        ]
        
        # Add reasoning parameter for o-family models
        if self.is_o_family:
            request_params["reasoning"] = {"effort": "medium"}

        response = self.client.responses.create(**request_params)
        return response.output_text
