"""
Dataset downloaders for different physical reasoning datasets.

This package provides centralized downloading functionality for all datasets
supported by PRKit. Each dataset has its own downloader that handles the
specific download mechanism (GitHub repos, HuggingFace, direct URLs, etc.).
"""

from .base_downloader import BaseDownloader
from .phybench_downloader import PHYBenchDownloader
from .physbench_downloader import PhysBenchDownloader
from .physics_downloader import PhysicsDownloader
from .physreason_downloader import PhysReasonDownloader
from .phyx_downloader import PhyXDownloader
from .seephys_downloader import SeePhysDownloader
from .ugphysics_downloader import UGPhysicsDownloader

__all__ = [
    "BaseDownloader",
    "PhysBenchDownloader",
    "PHYBenchDownloader",
    "PhysicsDownloader",
    "PhyXDownloader",
    "PhysReasonDownloader",
    "SeePhysDownloader",
    "UGPhysicsDownloader",
]
