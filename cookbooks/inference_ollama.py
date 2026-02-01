#!/usr/bin/env python3
"""
Quick Start: Run Inference with Ollama

This cookbook demonstrates how to:
- Check if Ollama is running
- Create an Ollama model client
- Run text-only inference
- Run vision inference with images (for vision models like qwen3-vl)

Prerequisites:
- physical-reasoning-toolkit installed (pip install physical-reasoning-toolkit)
- Ollama installed and running (https://ollama.com/download)
- Model pulled in Ollama: `ollama pull qwen3-vl`

Usage:
    python cookbooks/inference_ollama.py [model_name] [--prompt PROMPT] [--image IMAGE_PATH]
    
    
Examples:
    # Check if Ollama is running
    python cookbooks/inference_ollama.py --check
    
    # Text inference
    python cookbooks/inference_ollama.py qwen3 --prompt "What is quantum mechanics?"
    
    # Vision inference with image
    python cookbooks/inference_ollama.py qwen3-vl --prompt "Describe this image" --image cookbooks/data/sample.webp
"""

import argparse
import sys
from pathlib import Path

# Import the model client
from prkit.prkit_core.model_clients import create_model_client, OllamaModel
from prkit.prkit_core import PRKitLogger

# Set up logger
logger = PRKitLogger.get_logger(__name__)


def check_ollama_status():
    """Check if Ollama is running and show status."""
    logger.info("=" * 60)
    logger.info("Checking Ollama Status")
    logger.info("=" * 60)
    
    # Check if Ollama is running
    is_running = OllamaModel.check_ollama_running()
    
    if is_running:
        logger.info("✅ Ollama is running and accessible")
        
        # Try to list available models
        try:
            import ollama
            models = ollama.list()
            logger.info(f"\n📦 Available models ({len(models.get('models', []))}):")
            for model in models.get('models', []):
                model_name = model.get('name', 'Unknown')
                size = model.get('size', 0)
                size_gb = size / (1024**3) if size > 0 else 0
                logger.info(f"  • {model_name} ({size_gb:.2f} GB)")
        except Exception as e:
            logger.warning(f"Could not list models: {e}")
            logger.info("💡 Tip: Run `ollama list` in terminal to see available models")
        
        return True
    else:
        logger.error("❌ Ollama is not running or not accessible")
        logger.info("\n📋 To start Ollama:")
        logger.info("  1. Install Ollama from https://ollama.com/download")
        logger.info("  2. Start Ollama service (usually starts automatically)")
        logger.info("  3. Verify: `ollama list` or visit http://localhost:11434")
        logger.info("  4. Pull a model: `ollama pull qwen3-vl`")
        return False


def run_inference(model_name: str, prompt: str, image_path: str = None):
    """Run inference with Ollama model."""
    logger.info("=" * 60)
    logger.info("Ollama Inference")
    logger.info("=" * 60)
    
    # Check Ollama status first
    logger.info("\n🔍 Checking Ollama status...")
    if not OllamaModel.check_ollama_running():
        logger.error("Cannot proceed: Ollama is not running")
        sys.exit(1)
    logger.info("✅ Ollama is running")
    
    # Create model client
    logger.info(f"\n🤖 Creating client for model: {model_name}")
    try:
        client = create_model_client(model_name)
        logger.info(f"✅ Client created successfully (provider: {client.provider})")
    except ConnectionError as e:
        logger.error(f"❌ Connection error: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"❌ Model error: {e}")
        logger.info(f"\n💡 Tip: Pull the model first: `ollama pull {model_name}`")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Failed to create client: {e}")
        sys.exit(1)
    
    # Prepare image paths if provided
    image_paths = None
    if image_path:
        image_path = Path(image_path)
        if not image_path.exists():
            logger.error(f"❌ Image file not found: {image_path}")
            sys.exit(1)
        image_paths = [str(image_path)]
        logger.info(f"📷 Using image: {image_path}")
    
    # Run inference
    logger.info(f"\n💬 Running inference...")
    logger.info(f"   Prompt: {prompt}")
    if image_paths:
        logger.info(f"   Images: {len(image_paths)} image(s)")
    
    try:
        response = client.chat(user_prompt=prompt, image_paths=image_paths)
        
        logger.info("\n" + "=" * 60)
        logger.info("Response:")
        logger.info("=" * 60)
        logger.info(response)
        logger.info("=" * 60)
        logger.info("✅ Inference completed successfully!")
        
        return response
        
    except FileNotFoundError as e:
        logger.error(f"❌ File not found: {e}")
        sys.exit(1)
    except ConnectionError as e:
        logger.error(f"❌ Connection error during inference: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"❌ Model error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Inference failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run inference with Ollama models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check Ollama status
  python cookbooks/inference_ollama.py --check
  
  # Simple text inference
  python cookbooks/inference_ollama.py
  
  # Custom prompt
  python cookbooks/inference_ollama.py qwen3-vl --prompt "Explain quantum physics"
  
  # Vision inference
  python cookbooks/inference_ollama.py qwen3-vl --prompt "What's in this image?" --image image.jpg
        """
    )
    
    parser.add_argument(
        "model",
        nargs="?",
        default="qwen3-vl",
        help="Model name (default: qwen3-vl). Must be pulled in Ollama first.",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        default="What is physics?",
        help="Prompt text (default: 'What is physics?')",
    )
    parser.add_argument(
        "--image",
        "-i",
        type=str,
        default=None,
        help="Path to image file for vision inference",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check if Ollama is running, then exit",
    )
    
    args = parser.parse_args()
    
    # If --check flag is set, only check status
    if args.check:
        success = check_ollama_status()
        sys.exit(0 if success else 1)
    
    # Otherwise, run inference
    run_inference(
        model_name=args.model,
        prompt=args.prompt,
        image_path=args.image,
    )


if __name__ == "__main__":
    main()
