#!/usr/bin/env python3
"""
Cookbook 4: Environment Variables Demo

This cookbook demonstrates how to use the PHYSKIT_DATA_DIR environment variable
to customize where PhysKit looks for your datasets.

Prerequisites:
- physkit_datasets package installed
- Datasets downloaded to your preferred location

Usage:
    python 04_environment_variables_demo.py
"""

import os
import sys
from pathlib import Path

from prkit_datasets import DatasetHub

def main():
    """Main function demonstrating environment variable usage."""
    
    print("PhysKit Environment Variables Demo")
    print("=" * 40)
    
    # Show current environment
    print("Current Environment:")
    print(f"  PHYSKIT_DATA_DIR: {os.getenv('PHYSKIT_DATA_DIR', 'Not set')}")
    print(f"  HOME: {Path.home()}")
    print(f"  Default data location: {Path.home() / 'data'}")
    print()
    
    # Demonstrate different ways to set data directory
    
    print("1. Using PHYSKIT_DATA_DIR environment variable:")
    print("   Set this before running the script:")
    print("   export PHYSKIT_DATA_DIR='/Users/yinghuan/data'")
    print()
    
    print("2. Setting it in Python:")
    print("   os.environ['PHYSKIT_DATA_DIR'] = '/Users/yinghuan/data'")
    print()
    
    print("3. Using explicit data_dir parameter:")
    print("   dataset = DatasetHub.load('ugphysics', data_dir='/Users/yinghuan/data')")
    print()
    
    # Show priority order
    print("Priority Order for Data Directory Resolution:")
    print("  1. Explicit data_dir parameter (highest priority)")
    print("  2. PHYSKIT_DATA_DIR environment variable")
    print("  3. ~/data subdirectory (default)")
    print()
    
    # Test the environment variable
    print("Testing PHYSKIT_DATA_DIR Environment Variable:")
    print("-" * 45)
    
    # Set the environment variable for demonstration
    test_data_dir = "/Users/yinghuan/data"
    os.environ['PHYSKIT_DATA_DIR'] = test_data_dir
    print(f"✅ Set PHYSKIT_DATA_DIR = '{test_data_dir}'")
    
    try:
        # Now load dataset without specifying data_dir - should use environment variable
        dataset = DatasetHub.load("ugphysics")
        print(f"✅ Successfully loaded {len(dataset)} problems using PHYSKIT_DATA_DIR")
        print(f"   Environment variable worked correctly!")
        
    except Exception as e:
        print(f"❌ Error loading dataset with environment variable: {e}")
    
    # Clear the environment variable
    if 'PHYSKIT_DATA_DIR' in os.environ:
        del os.environ['PHYSKIT_DATA_DIR']
        print(f"🧹 Cleared PHYSKIT_DATA_DIR environment variable")
    
    print()

    
    print("Usage Examples:")
    print("  # Set for current session")
    print("  export PHYSKIT_DATA_DIR='/Users/yinghuan/data'")
    print("  python 04_environment_variables_demo.py")
    print()
    print("  # Set for specific command")
    print("  PHYSKIT_DATA_DIR='/Users/yinghuan/data' python 04_environment_variables_demo.py")
    print()
    print("  # Add to your shell profile (~/.bashrc, ~/.zshrc)")
    print("  echo 'export PHYSKIT_DATA_DIR=/Users/yinghuan/data' >> ~/.bashrc")
    print("  source ~/.bashrc")

if __name__ == "__main__":
    main()
