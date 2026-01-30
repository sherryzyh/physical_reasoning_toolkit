import json
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from ..logging_config import PhysKitLogger
from .physics_problem import PhysicsProblem


class PhysicalDataset:
    """
    Base class for physical reasoning datasets with a unified interface.

    This class now works directly with PhysicsProblem objects, providing
    a consistent interface across all PRKit (physical-reasoning-toolkit) packages.
    """

    def __init__(
        self,
        problems: List[PhysicsProblem],
        info: Optional[Dict[str, Any]] = None,
        split: str = "test",
    ):
        """
        Initialize dataset with PhysicsProblem objects.

        Args:
            problems: List of PhysicsProblem instances
            info: Optional dataset metadata
            split: Dataset split ("train", "test", "val", or "eval")
        """
        self._problems = problems
        self._info = info or {}
        self._split = split  # "train", "test", "val", or "eval"
        self._build_problem_id_index()

    def __len__(self) -> int:
        """Get the number of problems in the dataset."""
        return len(self._problems)

    def __getitem__(
        self, idx: Union[int, slice]
    ) -> Union[PhysicsProblem, "PhysicalDataset"]:
        """Get a problem by index or a slice of dataset."""
        if isinstance(idx, slice):
            # Return a new PhysicalDataset with sliced problems
            sliced_problems = [
                self._problems[i] for i in range(*idx.indices(len(self._problems)))
            ]
            return PhysicalDataset(sliced_problems, self._info, self._split)
        return self._problems[idx]

    def __iter__(self) -> Iterator[PhysicsProblem]:
        """Iterate over all problems."""
        return iter(self._problems)

    def _build_problem_id_index(self) -> None:
        """Build an internal index mapping problem_ids to problems for O(1) lookup."""
        self._problem_id_index = {}
        for i, problem in enumerate(self._problems):
            if hasattr(problem, "problem_id") and problem.problem_id:
                problem_id = problem.problem_id
                if problem_id in self._problem_id_index:
                    # Handle duplicate problem_ids by keeping track of the first occurrence
                    dataset_name = self._info.get("name", "unknown_dataset")
                    PhysKitLogger.get_logger(__name__).warning(
                        f"Dataset '{dataset_name}': Duplicate problem_id '{problem_id}' found. Using first occurrence."
                    )
                else:
                    self._problem_id_index[problem_id] = i
            else:
                # For problems without problem_id, use fallback
                fallback_id = f"problem_{i}"
                dataset_name = self._info.get("name", "unknown_dataset")
                PhysKitLogger.get_logger(__name__).warning(
                    f"Dataset '{dataset_name}': Problem at index {i} has no problem_id, using fallback '{fallback_id}'"
                )
                self._problem_id_index[fallback_id] = i

    def get_all_ids(self) -> List[str]:
        """Get all problem_ids in the dataset."""
        return list(self._problem_id_index.keys())

    def get_by_id(self, problem_id: str) -> PhysicsProblem:
        """
        Get a problem by problem_id using O(1) index lookup.

        Args:
            problem_id: The problem_id to search for

        Returns:
            PhysicsProblem with the matching problem_id

        Raises:
            KeyError: If no problem with the given problem_id is found
        """
        problem_index = self._problem_id_index[problem_id]
        return self._problems[problem_index]

    def get_by_id_safe(self, problem_id: str) -> Optional[PhysicsProblem]:
        """
        Get a problem by problem_id using O(1) index lookup, returning None if not found.

        Args:
            problem_id: The problem_id to search for

        Returns:
            PhysicsProblem with the matching problem_id, or None if not found
        """
        try:
            return self.get_by_id(problem_id)
        except KeyError:
            return None

    def filter(self, filter_func) -> "PhysicalDataset":
        """
        Filter problems using a filter function.

        Args:
            filter_func: Function that takes a PhysicsProblem and returns bool

        Returns:
            New PhysicalDataset with filtered problems
        """
        filtered_problems = [p for p in self._problems if filter_func(p)]
        return PhysicalDataset(filtered_problems, self._info, self._split)

    def filter_by_domains(
        self, domains: List[Union[str, "PhysicsDomain"]]
    ) -> "PhysicalDataset":
        """
        Filter problems by physics domains.

        Args:
            domains: List of domain names (strings) or PhysicsDomain enum values

        Returns:
            New PhysicalDataset containing only problems from the specified domains

        Example:
            # Filter by domain names
            mechanics_dataset = dataset.filter_by_domains(["mechanics", "classical_mechanics"])

            # Filter by PhysicsDomain enum values
            from prkit_core.definitions.physics_domain import PhysicsDomain
            quantum_dataset = dataset.filter_by_domains([PhysicsDomain.QUANTUM_MECHANICS])
        """
        # Import here to avoid circular imports
        from ..definitions.physics_domain import PhysicsDomain

        # Normalize domains to strings for comparison
        normalized_domains = set()
        for domain in domains:
            if isinstance(domain, PhysicsDomain):
                normalized_domains.add(domain.value)
            elif isinstance(domain, str):
                # Try to normalize the string domain
                try:
                    physics_domain = PhysicsDomain.from_string(domain)
                    normalized_domains.add(physics_domain.value)
                except (ValueError, AttributeError):
                    # If normalization fails, use the original string
                    normalized_domains.add(domain.lower())
            else:
                # Skip invalid domain types
                continue

        # Filter problems that match any of the specified domains
        filtered_problems = []
        for problem in self._problems:
            problem_domain = problem.get_domain_name()
            if problem_domain:
                # Check if the problem's domain exactly matches any of the specified domains
                if problem_domain.lower() in normalized_domains:
                    filtered_problems.append(problem)

        # Create new dataset with filtered problems
        filtered_dataset = PhysicalDataset(filtered_problems, self._info, self._split)

        # Log filtering results
        logger = PhysKitLogger.get_logger(__name__)
        logger.info(
            f"Filtered dataset by domains {list(normalized_domains)}: "
            f"{len(filtered_problems)} problems out of {len(self._problems)}"
        )

        return filtered_dataset

    def filter_by_domain(
        self, domain: Union[str, "PhysicsDomain"]
    ) -> "PhysicalDataset":
        """
        Filter problems by a single physics domain.

        Args:
            domain: Domain name (string) or PhysicsDomain enum value

        Returns:
            New PhysicalDataset containing only problems from the specified domain

        Example:
            # Filter by domain name
            mechanics_dataset = dataset.filter_by_domain("mechanics")

            # Filter by PhysicsDomain enum value
            from prkit_core.definitions.physics_domain import PhysicsDomain
            quantum_dataset = dataset.filter_by_domain(PhysicsDomain.QUANTUM_MECHANICS)
        """
        return self.filter_by_domains([domain])

    def select(self, indices: List[int]) -> "PhysicalDataset":
        """
        Select problems by indices.

        Args:
            indices: List of problem indices to select

        Returns:
            New PhysicalDataset with selected problems
        """
        selected_problems = [
            self._problems[i] for i in indices if 0 <= i < len(self._problems)
        ]
        return PhysicalDataset(selected_problems, self._info, self._split)

    def take(self, n: int) -> "PhysicalDataset":
        """
        Take the first N problems from the dataset.

        Args:
            n: Number of problems to take

        Returns:
            New PhysicalDataset with the first N problems
        """
        if n <= 0:
            return PhysicalDataset([], self._info, self._split)
        n = min(n, len(self._problems))
        return PhysicalDataset(self._problems[:n], self._info, self._split)

    def head(self, n: int = 5) -> "PhysicalDataset":
        """
        Get the first N problems (similar to pandas head).

        Args:
            n: Number of problems to get (default: 5)

        Returns:
            New PhysicalDataset with the first N problems
        """
        return self.take(n)

    def tail(self, n: int = 5) -> "PhysicalDataset":
        """
        Get the last N problems (similar to pandas tail).

        Args:
            n: Number of problems to get (default: 5)

        Returns:
            New PhysicalDataset with the last N problems
        """
        if n <= 0:
            return PhysicalDataset([], self._info, self._split)
        n = min(n, len(self._problems))
        return PhysicalDataset(self._problems[-n:], self._info, self._split)

    def sample(self, n: int) -> "PhysicalDataset":
        """
        Sample N problems from the dataset.

        Args:
            n: Number of problems to sample

        Returns:
            New PhysicalDataset with sampled problems
        """
        if n <= 0:
            return PhysicalDataset([], self._info, self._split)
        n = min(n, len(self._problems))
        return PhysicalDataset(
            random.sample(self._problems, n), self._info, self._split
        )

    def map(self, map_func) -> List[Any]:
        """
        Apply a function to each problem.

        Args:
            map_func: Function to apply to each PhysicsProblem

        Returns:
            List of results from applying map_func to each problem
        """
        return [map_func(p) for p in self._problems]

    def get_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        return self._info.copy()

    def get_split(self) -> str:
        """Get the dataset split."""
        return self._split

    @property
    def split(self) -> str:
        """Get the dataset split."""
        return self._split

    @property
    def name(self) -> str:
        """Get the dataset name."""
        return self._info.get("name", self.__class__.__name__)

    def to_list(self) -> List[Dict[str, Any]]:
        """Convert dataset to list of dictionaries."""
        return [problem.to_dict() for problem in self._problems]

    def save_to_json(self, filepath: Union[str, Path]) -> None:
        """Save dataset to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {"info": self._info, "split": self._split, "problems": self.to_list()}

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, filepath: Union[str, Path]) -> "PhysicalDataset":
        """Load dataset from JSON file."""
        filepath = Path(filepath)

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        problems = [PhysicsProblem.from_dict(p) for p in data["problems"]]
        return cls(problems, data.get("info"), data.get("split", "test"))

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if not self._problems:
            return {"total_problems": 0}

        # Domain distribution
        domain_counts = {}
        problem_types = {}
        languages = {}

        for problem in self._problems:
            # Count domains
            domain = problem.get_domain_name()
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

            # Count problem types
            problem_type = problem.problem_type
            problem_types[problem_type] = problem_types.get(problem_type, 0) + 1

            # Count languages
            language = problem.language
            languages[language] = languages.get(language, 0) + 1

        return {
            "total_problems": len(self._problems),
            "domains": domain_counts,
            "problem_types": problem_types,
            "languages": languages,
            "split": self._split,
        }

    def __repr__(self) -> str:
        return f"PhysicalDataset({len(self._problems)} problems, split='{self._split}')"

    def __str__(self) -> str:
        stats = self.get_statistics()
        return f"PhysicalDataset with {stats['total_problems']} problems ({stats['split']} split)"
