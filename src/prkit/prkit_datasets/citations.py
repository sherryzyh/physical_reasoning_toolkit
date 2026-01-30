"""
Centralized dataset citations for PRKit datasets.

This module provides citations for all supported datasets in a centralized location.
Citations are provided in BibTeX format and can be imported by loaders, downloaders,
and other components that need to reference the original papers.

Usage:
    from prkit.prkit_datasets.citations import get_citation, CITATIONS
    
    # Get citation for a specific dataset
    citation = get_citation("physreason")
    
    # Access all citations
    all_citations = CITATIONS
"""

from typing import Dict, Optional

# PhysReason citation
PHYSREASON_CITATION = """@inproceedings{zhang-etal-2025-physreason,
    title = "PhysReason: A Comprehensive Benchmark towards Physics-Based Reasoning",
    author = "Zhang, Xinyu  and
      Dong, Yuxuan  and
      Wu, Yanrui  and
      Huang, Jiaxing  and
      Jia, Chengyou  and
      Fernando, Basura  and
      Shou, Mike Zheng  and
      Zhang, Lingling  and
      Liu, Jun",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2025",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.811",
    pages = "1--20",
}"""

# SeePhys citation (placeholder - update when official citation is available)
SEEPHYS_CITATION = """@article{seephys,
    title = "SeePhys: A Visual Physics Reasoning Dataset",
    author = "SeePhys Team",
    journal = "OpenReview",
    year = "2024",
    url = "https://openreview.net/pdf?id=APNWmytTCS",
    note = "Paper available at https://openreview.net/pdf?id=APNWmytTCS",
}"""

# Dictionary mapping dataset names to their citations
CITATIONS: Dict[str, str] = {
    "physreason": PHYSREASON_CITATION,
    "seephys": SEEPHYS_CITATION,
}


def get_citation(dataset_name: str) -> Optional[str]:
    """
    Get the citation for a specific dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., "physreason", "seephys")
        
    Returns:
        BibTeX citation string, or None if dataset not found
        
    Examples:
        >>> citation = get_citation("physreason")
        >>> print(citation)
    """
    return CITATIONS.get(dataset_name.lower())


def list_cited_datasets() -> list:
    """
    List all datasets that have citations available.
    
    Returns:
        List of dataset names with citations
    """
    return list(CITATIONS.keys())
