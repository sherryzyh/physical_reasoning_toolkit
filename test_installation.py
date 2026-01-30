#!/usr/bin/env python3
"""
Test script to verify physical-reasoning-toolkit installation.

Run this after installing from TestPyPI or production PyPI to verify everything works.
"""

import sys
from typing import List, Tuple

def test_basic_imports() -> Tuple[bool, List[str]]:
    """Test basic package imports."""
    errors = []
    
    try:
        import prkit
        print(f"✅ prkit imported successfully (version: {prkit.__version__})")
    except ImportError as e:
        errors.append(f"❌ Failed to import prkit: {e}")
        return False, errors
    
    # Test package metadata
    try:
        assert prkit.__version__ == "0.1.0", f"Expected version 0.1.0, got {prkit.__version__}"
        assert prkit.__author__ == "Yinghuan Zhang"
        print(f"✅ Package metadata correct (author: {prkit.__author__})")
    except AssertionError as e:
        errors.append(f"❌ Package metadata mismatch: {e}")
    
    return len(errors) == 0, errors

def test_core_imports() -> Tuple[bool, List[str]]:
    """Test core package imports."""
    errors = []
    
    # Test package-level imports
    try:
        from prkit.prkit_core import PhysKitLogger
        print("✅ prkit.prkit_core.PhysKitLogger imported")
    except ImportError as e:
        errors.append(f"❌ Failed to import PhysKitLogger: {e}")
    
    try:
        from prkit.prkit_core.models import PhysicsProblem, PhysicalDataset
        print("✅ prkit.prkit_core.models imported")
    except ImportError as e:
        errors.append(f"❌ Failed to import models: {e}")
    
    try:
        from prkit.prkit_core.definitions import PhysicsDomain, AnswerType
        print("✅ prkit.prkit_core.definitions imported")
    except ImportError as e:
        errors.append(f"❌ Failed to import definitions: {e}")
    
    return len(errors) == 0, errors

def test_datasets_imports() -> Tuple[bool, List[str]]:
    """Test datasets package imports."""
    errors = []
    
    try:
        from prkit.prkit_datasets import DatasetHub
        print("✅ prkit.prkit_datasets.DatasetHub imported")
        
        # Test that DatasetHub has expected methods
        assert hasattr(DatasetHub, 'load'), "DatasetHub missing 'load' method"
        assert hasattr(DatasetHub, 'list_available'), "DatasetHub missing 'list_available' method"
        print("✅ DatasetHub has expected methods")
    except ImportError as e:
        errors.append(f"❌ Failed to import DatasetHub: {e}")
    except AssertionError as e:
        errors.append(f"❌ DatasetHub missing expected methods: {e}")
    
    return len(errors) == 0, errors

def test_evaluation_imports() -> Tuple[bool, List[str]]:
    """Test evaluation package imports."""
    errors = []
    
    try:
        from prkit.prkit_evaluation import AccuracyMetric
        print("✅ prkit.prkit_evaluation.AccuracyMetric imported")
    except ImportError as e:
        errors.append(f"❌ Failed to import AccuracyMetric: {e}")
    
    try:
        from prkit.prkit_evaluation.comparison import SmartAnswerComparator
        print("✅ prkit.prkit_evaluation.comparison imported")
    except ImportError as e:
        errors.append(f"❌ Failed to import SmartAnswerComparator: {e}")
    
    return len(errors) == 0, errors

def test_annotation_imports() -> Tuple[bool, List[str]]:
    """Test annotation package imports."""
    errors = []
    
    try:
        from prkit.prkit_annotation.workflows import WorkflowComposer
        print("✅ prkit.prkit_annotation.workflows.WorkflowComposer imported")
    except ImportError as e:
        errors.append(f"❌ Failed to import WorkflowComposer: {e}")
    
    return len(errors) == 0, errors

def test_top_level_imports() -> Tuple[bool, List[str]]:
    """Test top-level imports (after importing prkit)."""
    errors = []
    
    # Import prkit first to register subpackages
    import prkit
    
    try:
        from prkit_datasets import DatasetHub
        print("✅ Top-level import: prkit_datasets works")
    except ImportError as e:
        errors.append(f"❌ Top-level import prkit_datasets failed: {e}")
    
    try:
        from prkit_core import PhysKitLogger
        print("✅ Top-level import: prkit_core works")
    except ImportError as e:
        errors.append(f"❌ Top-level import prkit_core failed: {e}")
    
    try:
        from prkit_evaluation import AccuracyMetric
        print("✅ Top-level import: prkit_evaluation works")
    except ImportError as e:
        errors.append(f"❌ Top-level import prkit_evaluation failed: {e}")
    
    return len(errors) == 0, errors

def test_direct_imports() -> Tuple[bool, List[str]]:
    """Test direct imports from __init__.py."""
    errors = []
    
    try:
        import prkit
        # Test that main components are available directly
        assert hasattr(prkit, 'PhysKitLogger'), "PhysKitLogger not in prkit namespace"
        assert hasattr(prkit, 'PhysicsProblem'), "PhysicsProblem not in prkit namespace"
        assert hasattr(prkit, 'PhysicalDataset'), "PhysicalDataset not in prkit namespace"
        print("✅ Direct imports from prkit namespace work")
    except AssertionError as e:
        errors.append(f"❌ Direct imports failed: {e}")
    
    return len(errors) == 0, errors

def test_dependencies() -> Tuple[bool, List[str]]:
    """Test that required dependencies are available."""
    errors = []
    required_deps = [
        'pandas',
        'numpy',
        'openai',
        'pydantic',
        'tqdm',
        'sympy',
        'dotenv',  # python-dotenv
    ]
    
    for dep in required_deps:
        try:
            if dep == 'dotenv':
                __import__('dotenv')
            else:
                __import__(dep)
            print(f"✅ Dependency '{dep}' available")
        except ImportError:
            errors.append(f"❌ Required dependency '{dep}' not found")
    
    return len(errors) == 0, errors

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing physical-reasoning-toolkit Installation")
    print("=" * 60)
    print()
    
    all_passed = True
    all_errors = []
    
    # Run all test suites
    test_suites = [
        ("Basic Imports", test_basic_imports),
        ("Core Package", test_core_imports),
        ("Datasets Package", test_datasets_imports),
        ("Evaluation Package", test_evaluation_imports),
        ("Annotation Package", test_annotation_imports),
        ("Top-Level Imports", test_top_level_imports),
        ("Direct Imports", test_direct_imports),
        ("Dependencies", test_dependencies),
    ]
    
    for suite_name, test_func in test_suites:
        print(f"\n📦 Testing {suite_name}...")
        print("-" * 60)
        passed, errors = test_func()
        all_passed = all_passed and passed
        all_errors.extend(errors)
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    if all_passed:
        print("\n✅ All tests passed! Installation is working correctly.")
        return 0
    else:
        print(f"\n❌ {len(all_errors)} test(s) failed:")
        for error in all_errors:
            print(f"  {error}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
