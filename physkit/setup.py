#!/usr/bin/env python3
"""
Setup script for PhysKit - Physical Reasoning Toolkit
"""

from setuptools import setup, find_packages

# Read the README file for long description
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "Physical Reasoning Toolkit - Core Package"

# Read requirements
try:
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
except FileNotFoundError:
    requirements = []

setup(
    name="physkit",
    version="0.1.0",
    author="PhysKit Contributors",
    author_email="physkit@example.com",
    description="Physical Reasoning Toolkit - Core Package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sherryzyh/physical_reasoning_toolkit",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.12",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "myst-parser>=1.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "all": [
            "physkit-annotation",
            "physkit-datasets", 
            "physkit-evaluation",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.rst"],
    },
    entry_points={
        "console_scripts": [
            "physkit=physkit.cli:main",
        ],
    },
)
