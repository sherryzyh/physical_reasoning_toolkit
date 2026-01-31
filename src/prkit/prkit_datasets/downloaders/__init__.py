"""
Dataset downloaders for different physical reasoning datasets.

This package provides centralized downloading functionality for all datasets
supported by PRKit. Each dataset has its own downloader that handles the
specific download mechanism (GitHub repos, HuggingFace, direct URLs, etc.).
"""

from .base_downloader import BaseDownloader
from .phybench_downloader import PHYBenchDownloader
from .phyx_downloader import PhyXDownloader
from .physreason_downloader import PhysReasonDownloader
from .seephys_downloader import SeePhysDownloader
from .ugphysics_downloader import UGPhysicsDownloader

__all__ = [
    "BaseDownloader",
    "PHYBenchDownloader",
    "PhyXDownloader",
    "PhysReasonDownloader",
    "SeePhysDownloader",
    "UGPhysicsDownloader",
]
