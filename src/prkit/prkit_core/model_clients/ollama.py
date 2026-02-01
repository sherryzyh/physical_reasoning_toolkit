"""
Ollama API client implementation.

This module provides a client implementation for models hosted locally via Ollama.
It supports both text and vision models (like qwen3-vl, qwen2-vl, or llava).

Prerequisites:
- Ollama must be installed and running locally (default: http://localhost:11434)
- The model must be pulled first: `ollama pull qwen3-vl`
"""

import os
from typing import List, Optional
import ollama

from .base import BaseModelClient
from .utils import encode_image_to_base64


class OllamaModel(BaseModelClient):
    """Ollama model client implementation."""

    @staticmethod
    def check_ollama_running(base_url: Optional[str] = None) -> bool:
        """
        Check if Ollama service is running and accessible.
        
        Args:
            base_url: Optional base URL for Ollama API (defaults to http://localhost:11434)
            
        Returns:
            True if Ollama is running and accessible, False otherwise
        """
        try:
            if base_url:
                client = ollama.Client(host=base_url)
            else:
                client = ollama.Client()
            client.list()
            return True
        except Exception:
            return False

    def __init__(self, model: str, logger=None, base_url: Optional[str] = None):
        """
        Initialize Ollama model client.

        Args:
            model: The name of the model pulled in Ollama (e.g., 'qwen3-vl', 'qwen2.5', 'llava')
            logger: Optional logger instance
            base_url: Optional base URL for Ollama API (defaults to http://localhost:11434)
            
        Raises:
            ConnectionError: If Ollama service is not running or unreachable
        """
        super().__init__(model, logger)
        self.provider = "ollama"
        self.base_url = base_url
        # The ollama-python library uses a default client pointing to localhost:11434
        # but you can also use ollama.Client(host='...') if needed.
        
        # Check if Ollama is running during initialization
        self._check_ollama_running()

    def _check_ollama_running(self):
        """
        Check if Ollama service is running and accessible.
        
        Raises:
            ConnectionError: If Ollama service is not running or unreachable
        """
        try:
            # Try to list models as a simple connectivity check
            if self.base_url:
                client = ollama.Client(host=self.base_url)
            else:
                client = ollama.Client()
            client.list()
        except Exception as e:
            error_msg = (
                f"Ollama service is not running or unreachable at "
                f"{self.base_url or 'http://localhost:11434'}. "
                f"Please ensure Ollama is installed and running.\n"
                f"To start Ollama:\n"
                f"  1. Install Ollama from https://ollama.com/download\n"
                f"  2. Start the Ollama service (usually starts automatically after installation)\n"
                f"  3. Verify it's running: `ollama list` or visit http://localhost:11434\n"
                f"Original error: {str(e)}"
            )
            self.logger.error(error_msg)
            raise ConnectionError(error_msg) from e

    def chat(self, user_prompt: str, image_paths: Optional[List[str]] = None) -> str:
        """
        Generate a response using the local Ollama service.

        Args:
            user_prompt: The user's prompt text.
            image_paths: Optional list of file paths to images. 
                         Ollama accepts local file paths.

        Returns:
            Response text from the model.
            
        Raises:
            FileNotFoundError: If any image file path doesn't exist
            ConnectionError: If Ollama service becomes unreachable during inference
            ValueError: If the model is not found (not pulled in Ollama)
        """
        message = {
            'role': 'user',
            'content': user_prompt,
        }

        if image_paths:
            # Ollama's python SDK can handle paths, but for consistency with your 
            # OpenAI implementation, we will ensure they exist.
            valid_images = []
            for path in image_paths:
                if not os.path.exists(path):
                    self.logger.error(f"Image not found: {path}")
                    raise FileNotFoundError(f"Image file not found: {path}")
                valid_images.append(path)
            
            message['images'] = valid_images

        try:
            # Use Client if base_url is specified, otherwise use default
            if self.base_url:
                client = ollama.Client(host=self.base_url)
                response = client.chat(
                    model=self.model,
                    messages=[message],
                    options={
                        'temperature': 0.7,
                    }
                )
            else:
                response = ollama.chat(
                    model=self.model,
                    messages=[message],
                    options={
                        'temperature': 0.7,
                    }
                )
            
            # Handle both dict-like and object-like response access
            if hasattr(response, "message"):
                return response.message.content
            else:
                return response['message']['content']
            
        except Exception as e:
            # Check if it's a model not found error (ResponseError with 404 or model not found message)
            error_str = str(e).lower()
            error_type = type(e).__name__
            
            # Handle model not found errors
            if ("model" in error_str and "not found" in error_str) or \
               (hasattr(e, 'status_code') and getattr(e, 'status_code') == 404) or \
               (error_type == 'ResponseError' and "404" in error_str):
                error_msg = (
                    f"Model '{self.model}' not found in Ollama. "
                    f"Please pull the model first: `ollama pull {self.model}`\n"
                    f"To list available models: `ollama list`\n"
                    f"Original error: {str(e)}"
                )
                self.logger.error(error_msg)
                raise ValueError(error_msg) from e
            
            # Handle connection errors
            if "connection" in error_str or "refused" in error_str or "unreachable" in error_str or \
               (hasattr(e, 'status_code') and getattr(e, 'status_code', 0) >= 500):
                error_msg = (
                    f"Ollama service is not running or unreachable at "
                    f"{self.base_url or 'http://localhost:11434'}. "
                    f"Please ensure Ollama is running.\n"
                    f"To check: `ollama list` or visit http://localhost:11434\n"
                    f"Original error: {str(e)}"
                )
                self.logger.error(error_msg)
                raise ConnectionError(error_msg) from e
            
            # Re-raise other exceptions as-is
            self.logger.error(f"Ollama inference failed: {str(e)}")
            raise