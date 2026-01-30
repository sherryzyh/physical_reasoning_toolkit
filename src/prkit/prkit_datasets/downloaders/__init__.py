"""
Dataset downloaders for different physical reasoning datasets.

This package provides centralized downloading functionality for all datasets
supported by PRKit. Each dataset has its own downloader that handles the
specific download mechanism (GitHub repos, HuggingFace, direct URLs, etc.).
"""

__version__ = "0.1.0"
__author__ = "Yinghuan Zhang"
__author_email__ = "yinghuan.flash@gmail.com"

from .base_downloader import BaseDownloader
from .physreason_downloader import PhysReasonDownloader
from .seephys_downloader import SeePhysDownloader

__all__ = [
    "BaseDownloader",
    "PhysReasonDownloader",
    "SeePhysDownloader",
]
