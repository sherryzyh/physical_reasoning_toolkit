#!/usr/bin/env python3
"""
Quick Start: Run Inference with OpenAI

This cookbook demonstrates how to:
- Check if OpenAI API key is configured
- Create an OpenAI model client
- Run text-only inference
- Run vision inference with images (for supported models like gpt-4.1, gpt-5xxxx, o-family)

Prerequisites:
- physical-reasoning-toolkit installed (pip install physical-reasoning-toolkit)
- OpenAI API key set in environment: `export OPENAI_API_KEY=your_key_here`
- OpenAI SDK installed: `pip install openai`

Usage:
    python cookbooks/inference_openai.py [model_name] [--prompt PROMPT] [--image IMAGE_PATH]
    
    
Examples:
    # Check if OpenAI API key is configured
    python cookbooks/inference_openai.py --check
    
    # Text inference with default model
    python cookbooks/inference_openai.py --prompt "What is quantum mechanics?"
    
    # Text inference with specific model
    python cookbooks/inference_openai.py o3-mini --prompt "What is the formula for kinetic energy?"
    
    # Vision inference with image (for supported models)
    python cookbooks/inference_openai.py gpt-5.1 --prompt "Describe this image" --image cookbooks/data/sample.webp
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Import the model client
from prkit.prkit_core.model_clients import create_model_client
from prkit.prkit_core import PRKitLogger

# Load environment variables from .env file
load_dotenv()

# Set up logger
logger = PRKitLogger.get_logger(__name__)


def check_openai_status():
    """Check if OpenAI API key is configured and show status."""
    logger.info("=" * 60)
    logger.info("Checking OpenAI Configuration")
    logger.info("=" * 60)
    
    # Check if API key is set
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if api_key:
        # Mask the key for display
        masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        logger.info(f"✅ OpenAI API key is configured ({masked_key})")
        
        # Try to verify OpenAI SDK is available
        try:
            from openai import OpenAI  # noqa: F401
            logger.info("✅ OpenAI SDK is available")
            logger.info("\n📋 Supported OpenAI models:")
            logger.info("  • gpt-4.1 (and variants: gpt-4.1-mini, gpt-4.1-nano)")
            logger.info("  • gpt-5xxxx (gpt-5, gpt-5.1, gpt-5.2, gpt-5.1-mini, etc.)")
            logger.info("  • o-family (o3, o4, o4-mini, etc. - reasoning models)")
            logger.info("\n💡 Note: Vision support is available for all OpenAI models")
            return True
        except ImportError:
            logger.error("❌ OpenAI SDK not installed")
            logger.info("\n📋 To install: `pip install openai`")
            return False
        except Exception as e:
            logger.warning(f"⚠️  Could not verify OpenAI connection: {e}")
            logger.info("💡 This might be normal if you haven't made any API calls yet")
            return True
    else:
        logger.error("❌ OpenAI API key is not configured")
        logger.info("\n📋 To set up OpenAI:")
        logger.info("  1. Get an API key from https://platform.openai.com/api-keys")
        logger.info("  2. Set environment variable: `export OPENAI_API_KEY=your_key_here`")
        logger.info("  3. Or add to .env file: `OPENAI_API_KEY=your_key_here`")
        logger.info("  4. Install OpenAI SDK: `pip install openai`")
        return False


def run_inference(model_name: str, prompt: str, image_path: str = None):
    """Run inference with OpenAI model."""
    logger.info("=" * 60)
    logger.info("OpenAI Inference")
    logger.info("=" * 60)
    
    # Check OpenAI status first
    logger.info("\n🔍 Checking OpenAI configuration...")
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("Cannot proceed: OPENAI_API_KEY is not set")
        logger.info("💡 Set it with: `export OPENAI_API_KEY=your_key_here`")
        sys.exit(1)
    logger.info("✅ OpenAI API key is configured")
    
    # Create model client
    logger.info(f"\n🤖 Creating client for model: {model_name}")
    try:
        client = create_model_client(model_name)
        logger.info(f"✅ Client created successfully (provider: {client.provider})")
    except ValueError as e:
        logger.error(f"❌ Model error: {e}")
        logger.info(f"\n💡 Supported models: gpt-4.1, gpt-5xxxx, o-family (o3, o4, etc.)")
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
    except Exception as e:
        logger.error(f"❌ Inference failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run inference with OpenAI models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check OpenAI configuration
  python cookbooks/inference_openai.py --check
  
  # Simple text inference with default model
  python cookbooks/inference_openai.py --prompt "What is physics?"
  
  # Custom model and prompt
  python cookbooks/inference_openai.py gpt-4.1 --prompt "Explain quantum physics"
  
  # Vision inference
  python cookbooks/inference_openai.py gpt-4.1 --prompt "What's in this image?" --image image.jpg
  
  # Using o-family reasoning models
  python cookbooks/inference_openai.py o3-mini --prompt "Solve this problem"
        """
    )
    
    parser.add_argument(
        "model",
        nargs="?",
        default="gpt-4.1",
        help="Model name (default: gpt-4.1). Supported: gpt-4.1, gpt-5xxxx, o-family (o3, o4, etc.)",
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
        help="Only check if OpenAI is configured, then exit",
    )
    
    args = parser.parse_args()
    
    # If --check flag is set, only check status
    if args.check:
        success = check_openai_status()
        sys.exit(0 if success else 1)
    
    # Otherwise, run inference
    run_inference(
        model_name=args.model,
        prompt=args.prompt,
        image_path=args.image,
    )


if __name__ == "__main__":
    main()
