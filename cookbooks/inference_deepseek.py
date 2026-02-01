#!/usr/bin/env python3
"""
Quick Start: Run Inference with DeepSeek

This cookbook demonstrates how to:
- Check if DeepSeek API key is configured
- Create a DeepSeek model client
- Run text-only inference
- Note: DeepSeek models do not support vision

Prerequisites:
- physical-reasoning-toolkit installed (pip install physical-reasoning-toolkit)
- DeepSeek API key set in environment: `export DEEPSEEK_API_KEY=your_key_here`
- OpenAI SDK installed: `pip install openai` (DeepSeek uses OpenAI-compatible API)

Usage:
    python cookbooks/inference_deepseek.py [model_name] [--prompt PROMPT]
    
    
Examples:
    # Check if DeepSeek API key is configured
    python cookbooks/inference_deepseek.py --check
    
    # Text inference with default model
    python cookbooks/inference_deepseek.py --prompt "What is quantum mechanics?"
    
    # Text inference with specific model
    python cookbooks/inference_deepseek.py deepseek-chat --prompt "Explain quantum physics"
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


def check_deepseek_status():
    """Check if DeepSeek API key is configured and show status."""
    logger.info("=" * 60)
    logger.info("Checking DeepSeek Configuration")
    logger.info("=" * 60)
    
    # Check if API key is set
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    
    if api_key:
        # Mask the key for display
        masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        logger.info(f"✅ DeepSeek API key is configured ({masked_key})")
        
        # Try to verify OpenAI SDK is available (DeepSeek uses OpenAI-compatible API)
        try:
            from openai import OpenAI  # noqa: F401
            logger.info("✅ OpenAI SDK is available (required for DeepSeek)")
            logger.info("\n📋 Supported DeepSeek models:")
            logger.info("  • deepseek-chat (default, non-thinking mode)")
            logger.info("  • deepseek-reasoner (thinking mode)")
            logger.info("  • (and other deepseek-* models)")
            logger.info("\n⚠️  Note: DeepSeek models do not support vision")
            logger.info("   Images will be ignored with a warning if provided")
            return True
        except ImportError:
            logger.error("❌ OpenAI SDK not installed")
            logger.info("\n📋 To install: `pip install openai`")
            return False
        except Exception as e:
            logger.warning(f"⚠️  Could not verify DeepSeek connection: {e}")
            logger.info("💡 This might be normal if you haven't made any API calls yet")
            return True
    else:
        logger.error("❌ DeepSeek API key is not configured")
        logger.info("\n📋 To set up DeepSeek:")
        logger.info("  1. Get an API key from https://platform.deepseek.com/api_keys")
        logger.info("  2. Set environment variable: `export DEEPSEEK_API_KEY=your_key_here`")
        logger.info("  3. Or add to .env file: `DEEPSEEK_API_KEY=your_key_here`")
        logger.info("  4. Install OpenAI SDK: `pip install openai`")
        return False


def run_inference(model_name: str, prompt: str, image_path: str = None):
    """Run inference with DeepSeek model."""
    logger.info("=" * 60)
    logger.info("DeepSeek Inference")
    logger.info("=" * 60)
    
    # Check DeepSeek status first
    logger.info("\n🔍 Checking DeepSeek configuration...")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        logger.error("Cannot proceed: DEEPSEEK_API_KEY is not set")
        logger.info("💡 Set it with: `export DEEPSEEK_API_KEY=your_key_here`")
        sys.exit(1)
    logger.info("✅ DeepSeek API key is configured")
    
    # Warn about image support
    if image_path:
        logger.warning("\n⚠️  Note: DeepSeek models do not support vision")
        logger.warning("   Images will be ignored. Only text inference will be performed.")
    
    # Create model client
    logger.info(f"\n🤖 Creating client for model: {model_name}")
    try:
        client = create_model_client(model_name)
        logger.info(f"✅ Client created successfully (provider: {client.provider})")
    except ValueError as e:
        logger.error(f"❌ Model error: {e}")
        logger.info(f"\n💡 Supported models: deepseek-chat, deepseek-reasoner")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Failed to create client: {e}")
        sys.exit(1)
    
    # Run inference
    logger.info(f"\n💬 Running inference...")
    logger.info(f"   Prompt: {prompt}")
    if image_path:
        logger.info(f"   ⚠️  Image provided but will be ignored (vision not supported)")
    
    try:
        # Note: image_paths parameter is accepted but images will be ignored
        response = client.chat(user_prompt=prompt, image_paths=[image_path] if image_path else None)
        
        logger.info("\n" + "=" * 60)
        logger.info("Response:")
        logger.info("=" * 60)
        logger.info(response)
        logger.info("=" * 60)
        logger.info("✅ Inference completed successfully!")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Inference failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run inference with DeepSeek models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check DeepSeek configuration
  python cookbooks/inference_deepseek.py --check
  
  # Simple text inference with default model
  python cookbooks/inference_deepseek.py --prompt "What is physics?"
  
  # Custom model and prompt
  python cookbooks/inference_deepseek.py deepseek-chat --prompt "Explain quantum physics"
  
  # Note: Vision inference is not supported
  # Images will be ignored if provided
        """
    )
    
    parser.add_argument(
        "model",
        nargs="?",
        default="deepseek-chat",
        help="Model name (default: deepseek-chat). Supported: deepseek-chat, deepseek-reasoner",
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
        help="Path to image file (currently ignored - vision not supported)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check if DeepSeek is configured, then exit",
    )
    
    args = parser.parse_args()
    
    # If --check flag is set, only check status
    if args.check:
        success = check_deepseek_status()
        sys.exit(0 if success else 1)
    
    # Otherwise, run inference
    run_inference(
        model_name=args.model,
        prompt=args.prompt,
        image_path=args.image,
    )


if __name__ == "__main__":
    main()
