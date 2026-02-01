#!/usr/bin/env python3
"""
Quick Start: Run Inference with Google Gemini

This cookbook demonstrates how to:
- Check if Gemini API key is configured
- Create a Gemini model client
- Run text-only inference
- Note: Vision support is planned but not yet implemented

Prerequisites:
- physical-reasoning-toolkit installed (pip install physical-reasoning-toolkit)
- Google Gemini API key set in environment: `export GEMINI_API_KEY=your_key_here`
  (or `export GOOGLE_API_KEY=your_key_here` for backward compatibility)
- Google Generative AI SDK installed: `pip install google-genai`

Usage:
    python cookbooks/inference_gemini.py [model_name] [--prompt PROMPT]
    
    
Examples:
    # Check if Gemini API key is configured
    python cookbooks/inference_gemini.py --check
    
    # Text inference with default model
    python cookbooks/inference_gemini.py --prompt "What is quantum mechanics?"
    
    # Text inference with specific model
    python cookbooks/inference_gemini.py gemini-2.5-pro --prompt "Explain quantum physics"
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


def check_gemini_status():
    """Check if Gemini API key is configured and show status."""
    logger.info("=" * 60)
    logger.info("Checking Gemini Configuration")
    logger.info("=" * 60)
    
    # Check if API key is set (support both GEMINI_API_KEY and GOOGLE_API_KEY)
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    
    if api_key:
        # Mask the key for display
        masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        key_var = "GEMINI_API_KEY" if os.environ.get("GEMINI_API_KEY") else "GOOGLE_API_KEY"
        logger.info(f"✅ Gemini API key is configured ({key_var}: {masked_key})")
        
        # Try to verify Google Generative AI SDK is available
        try:
            from google import genai  # noqa: F401
            logger.info("✅ Google Generative AI SDK is available")
            logger.info("\n📋 Supported Gemini models:")
            logger.info("  • gemini-2.5-flash (default, latest)")
            logger.info("  • gemini-1.5-pro (more capable)")
            logger.info("  • gemini-1.5-flash (faster, cost-efficient)")
            logger.info("  • gemini-1.5-pro-002")
            logger.info("  • gemini-1.5-flash-002")
            logger.info("\n⚠️  Note: Vision support is planned but not yet implemented")
            logger.info("   Images will be ignored with a warning if provided")
            return True
        except ImportError:
            logger.error("❌ Google Generative AI SDK not installed")
            logger.info("\n📋 To install: `pip install google-genai`")
            return False
        except Exception as e:
            logger.warning(f"⚠️  Could not verify Gemini connection: {e}")
            logger.info("💡 This might be normal if you haven't made any API calls yet")
            return True
    else:
        logger.error("❌ Gemini API key is not configured")
        logger.info("\n📋 To set up Gemini:")
        logger.info("  1. Get an API key from https://aistudio.google.com/app/apikey")
        logger.info("  2. Set environment variable: `export GEMINI_API_KEY=your_key_here`")
        logger.info("     (or `export GOOGLE_API_KEY=your_key_here` for backward compatibility)")
        logger.info("  3. Or add to .env file: `GEMINI_API_KEY=your_key_here`")
        logger.info("  4. Install Google Generative AI SDK: `pip install google-genai`")
        return False


def run_inference(model_name: str, prompt: str, image_path: str = None):
    """Run inference with Gemini model."""
    logger.info("=" * 60)
    logger.info("Gemini Inference")
    logger.info("=" * 60)
    
    # Check Gemini status first
    logger.info("\n🔍 Checking Gemini configuration...")
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error("Cannot proceed: GEMINI_API_KEY or GOOGLE_API_KEY is not set")
        logger.info("💡 Set it with: `export GEMINI_API_KEY=your_key_here`")
        sys.exit(1)
    logger.info("✅ Gemini API key is configured")
    
    # Warn about image support
    if image_path:
        logger.warning("\n⚠️  Note: Vision support is not yet implemented for Gemini models")
        logger.warning("   Images will be ignored. Only text inference will be performed.")
    
    # Create model client
    logger.info(f"\n🤖 Creating client for model: {model_name}")
    try:
        client = create_model_client(model_name)
        logger.info(f"✅ Client created successfully (provider: {client.provider})")
    except ValueError as e:
        logger.error(f"❌ Model error: {e}")
        logger.info(f"\n💡 Supported models: gemini-2.5-flash, gemini-1.5-pro, gemini-1.5-flash, gemini-1.5-pro-002, gemini-1.5-flash-002")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Failed to create client: {e}")
        sys.exit(1)
    
    # Run inference
    logger.info(f"\n💬 Running inference...")
    logger.info(f"   Prompt: {prompt}")
    if image_path:
        logger.info(f"   ⚠️  Image provided but will be ignored (vision not yet implemented)")
    
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
        description="Run inference with Google Gemini models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check Gemini configuration
  python cookbooks/inference_gemini.py --check
  
  # Simple text inference with default model
  python cookbooks/inference_gemini.py --prompt "What is physics?"
  
  # Custom model and prompt
  python cookbooks/inference_gemini.py gemini-1.5-pro --prompt "Explain quantum physics"
  
  # Note: Vision inference is not yet supported
  # Images will be ignored if provided
        """
    )
    
    parser.add_argument(
        "model",
        nargs="?",
        default="gemini-2.5-flash",
        help="Model name (default: gemini-2.5-flash). Supported: gemini-1.5-pro, gemini-1.5-flash, gemini-2.5-flash, etc.",
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
        help="Path to image file (currently ignored - vision support planned)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check if Gemini is configured, then exit",
    )
    
    args = parser.parse_args()
    
    # If --check flag is set, only check status
    if args.check:
        success = check_gemini_status()
        sys.exit(0 if success else 1)
    
    # Otherwise, run inference
    run_inference(
        model_name=args.model,
        prompt=args.prompt,
        image_path=args.image,
    )


if __name__ == "__main__":
    main()
