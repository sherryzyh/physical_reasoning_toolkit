# Dataset Downloaders

This package provides centralized downloading functionality for all physical reasoning datasets supported by PRKit.

## Architecture

The downloaders package follows the same pattern as the `loaders/` package:

- **`base_downloader.py`**: Abstract base class defining the downloader interface
- **Individual downloaders**: One downloader class per dataset (e.g., `ugphysics_downloader.py`, `phybench_downloader.py`)

## Base Downloader Interface

All downloaders inherit from `BaseDownloader` and implement:

```python
class MyDatasetDownloader(BaseDownloader):
    @property
    def dataset_name(self) -> str:
        """Return the dataset name."""
        return "mydataset"
    
    @property
    def download_info(self) -> Dict[str, Any]:
        """Return download metadata."""
        return {
            "source": "GitHub repository",
            "url": "https://github.com/...",
            "size_bytes": 1024 * 1024 * 100,  # Optional
            "format": "jsonl",
        }
    
    def download(self, data_dir=None, force=False, **kwargs) -> Path:
        """Download the dataset."""
        pass
    
    def verify(self, data_dir: Path) -> bool:
        """Verify the downloaded dataset."""
        pass
```

## Usage

### Direct Usage

```python
from prkit.prkit_datasets.downloaders import UGPhysicsDownloader

downloader = UGPhysicsDownloader()
download_path = downloader.download(force=False)
print(f"Dataset downloaded to: {download_path}")
```

### Integration with DatasetHub

The downloaders can be integrated with `DatasetHub` to provide automatic downloading:

```python
from prkit.prkit_datasets import DatasetHub

# This could automatically download if dataset doesn't exist
dataset = DatasetHub.load("ugphysics", auto_download=True)
```

## Download Sources

Different datasets may come from different sources:

- **GitHub Repositories**: Clone or download from GitHub
- **HuggingFace**: Use HuggingFace datasets library
- **Direct URLs**: Download from direct file URLs
- **Google Drive / Dropbox**: Download from cloud storage
- **Academic Repositories**: Download from research paper repositories

Each downloader handles the specific mechanism for its dataset.

## Directory Structure

Downloaded datasets are stored in:

- `DATASET_CACHE_DIR/{dataset_name}/` (if `DATASET_CACHE_DIR` is set)
- `~/PHYSICAL_REASONING_DATASETS/{dataset_name}/` (default fallback)

This matches the directory structure expected by the loaders.

## Implementation Checklist

For each dataset, create a downloader that:

1. ✅ Inherits from `BaseDownloader`
2. ✅ Implements `dataset_name` property
3. ✅ Implements `download_info` property with source information
4. ✅ Implements `download()` method with progress tracking
5. ✅ Implements `verify()` method to check download completeness
6. ✅ Handles errors gracefully with informative messages
7. ✅ Supports `force` parameter to re-download existing datasets
8. ✅ Uses `resolve_download_dir()` for consistent directory resolution
