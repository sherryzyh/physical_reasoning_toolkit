"""
Tests for dataset citations module.
"""

from prkit.prkit_datasets.citations import (
    CITATIONS,
    get_citation,
    list_cited_datasets,
)


class TestCitations:
    """Test cases for citations module."""

    def test_citations_dict_not_empty(self):
        """Test that CITATIONS dictionary is not empty."""
        assert len(CITATIONS) > 0
        assert isinstance(CITATIONS, dict)

    def test_get_citation_existing(self):
        """Test getting citation for existing dataset."""
        citation = get_citation("physreason")
        assert citation is not None
        assert isinstance(citation, str)
        assert "@inproceedings" in citation or "@article" in citation or "@misc" in citation

    def test_get_citation_phyx(self):
        """Test getting citation for PhyX dataset."""
        citation = get_citation("phyx")
        assert citation is not None
        assert isinstance(citation, str)
        assert "@misc" in citation or "@article" in citation
        assert "PhyX" in citation or "phyx" in citation.lower()

    def test_get_citation_case_insensitive(self):
        """Test that get_citation is case insensitive."""
        citation1 = get_citation("physreason")
        citation2 = get_citation("PHYSREASON")
        citation3 = get_citation("PhysReason")

        assert citation1 == citation2 == citation3

    def test_get_citation_nonexistent(self):
        """Test getting citation for nonexistent dataset returns None."""
        citation = get_citation("nonexistent_dataset")
        assert citation is None

    def test_list_cited_datasets(self):
        """Test listing all cited datasets."""
        datasets = list_cited_datasets()
        assert isinstance(datasets, list)
        assert len(datasets) > 0

        # Check that known datasets are in the list
        dataset_names_lower = [d.lower() for d in datasets]
        assert "physreason" in dataset_names_lower or any(
            "physreason" in d.lower() for d in datasets
        )

    def test_citations_contain_expected_datasets(self):
        """Test that citations contain expected dataset names."""
        # Check for known datasets
        citation_keys = list(CITATIONS.keys())
        citation_keys_lower = [k.lower() for k in citation_keys]

        # At least one of these should be present
        expected_datasets = ["physreason", "seephys", "phybench", "ugphysics", "phyx"]
        assert any(ds in citation_keys_lower for ds in expected_datasets)

    def test_citations_are_bibtex_format(self):
        """Test that citations are in BibTeX format."""
        for _dataset_name, citation in CITATIONS.items():
            assert isinstance(citation, str)
            # BibTeX entries typically start with @
            assert citation.strip().startswith("@")
            # Should contain title or other BibTeX fields
            assert "title" in citation.lower() or "author" in citation.lower()

    def test_get_citation_all_datasets(self):
        """Test getting citations for all available datasets."""
        datasets = list_cited_datasets()
        for dataset in datasets:
            citation = get_citation(dataset)
            assert citation is not None, f"Citation not found for {dataset}"
            assert len(citation) > 0
