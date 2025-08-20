from typing import Dict, Any, List, Optional, Union, Iterator
from pathlib import Path
import json

from .physics_problem import PhysicsProblem
from ..logging_config import PhysKitLogger


class PhysicalDataset:
    """
    Base class for physical reasoning datasets with a unified interface.
    
    This class now works directly with PhysicsProblem objects, providing
    a consistent interface across all PhysKit packages.
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
    
    def __getitem__(self, idx: Union[int, slice]) -> Union[PhysicsProblem, List[PhysicsProblem]]:
        """Get problem(s) by index."""
        if isinstance(idx, slice):
            return [self._problems[i] for i in range(*idx.indices(len(self._problems)))]
        return self._problems[idx]
    
    def __iter__(self) -> Iterator[PhysicsProblem]:
        """Iterate over all problems."""
        return iter(self._problems)
    
    def _build_problem_id_index(self) -> None:
        """Build an internal index mapping problem_ids to problems for O(1) lookup."""
        self._problem_id_index = {}
        for i, problem in enumerate(self._problems):
            if hasattr(problem, 'problem_id') and problem.problem_id:
                problem_id = problem.problem_id
                if problem_id in self._problem_id_index:
                    # Handle duplicate problem_ids by keeping track of the first occurrence
                    PhysKitLogger.get_logger(__name__).warning(f"Duplicate problem_id '{problem_id}' found. Using first occurrence.")
                else:
                    self._problem_id_index[problem_id] = i
            else:
                # For problems without problem_id, use fallback
                fallback_id = f'problem_{i}'
                self._problem_id_index[fallback_id] = i
    
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
    
    def filter(self, filter_func) -> 'PhysicalDataset':
        """
        Filter problems using a filter function.
        
        Args:
            filter_func: Function that takes a PhysicsProblem and returns bool
            
        Returns:
            New PhysicalDataset with filtered problems
        """
        filtered_problems = [p for p in self._problems if filter_func(p)]
        return PhysicalDataset(filtered_problems, self._info, self._split)
    
    def select(self, indices: List[int]) -> 'PhysicalDataset':
        """
        Select problems by indices.
        
        Args:
            indices: List of problem indices to select
            
        Returns:
            New PhysicalDataset with selected problems
        """
        selected_problems = [self._problems[i] for i in indices if 0 <= i < len(self._problems)]
        return PhysicalDataset(selected_problems, self._info, self._split)
    
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
    
    def to_list(self) -> List[Dict[str, Any]]:
        """Convert dataset to list of dictionaries."""
        return [problem.to_dict() for problem in self._problems]
    
    def save_to_json(self, filepath: Union[str, Path]) -> None:
        """Save dataset to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'info': self._info,
            'split': self._split,
            'problems': self.to_list()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def from_json(cls, filepath: Union[str, Path]) -> 'PhysicalDataset':
        """Load dataset from JSON file."""
        filepath = Path(filepath)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        problems = [PhysicsProblem.from_dict(p) for p in data['problems']]
        return cls(problems, data.get('info'), data.get('split', 'test'))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if not self._problems:
            return {'total_problems': 0}
        
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
            'total_problems': len(self._problems),
            'domains': domain_counts,
            'problem_types': problem_types,
            'languages': languages,
            'split': self._split
        }
    
    def __repr__(self) -> str:
        return f"PhysicalDataset({len(self._problems)} problems, split='{self._split}')"
    
    def __str__(self) -> str:
        stats = self.get_statistics()
        return f"PhysicalDataset with {stats['total_problems']} problems ({stats['split']} split)"



