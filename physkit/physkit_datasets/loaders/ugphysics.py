"""
UGPhysics Dataset Loader

This module provides a loader for the UGPhysics dataset, which contains
undergraduate physics problems across multiple domains.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Union, Optional
import random # Added for per_domain sampling

from physkit.models import PhysicsDomain, PhysicalDataset
from physkit_datasets.loaders.base_loader import BaseDatasetLoader


class UGPhysicsLoader(BaseDatasetLoader):
    """Loader for UGPhysics dataset."""

    @property
    def name(self) -> str:
        """Return the name of the dataset."""
        return "ugphysics"
    
    @property
    def description(self) -> str:
        """Return the description of the dataset."""
        return "Undergraduate Physics dataset with problems across multiple domains"
    
    def get_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        return {
            "name": self.name,
            "description": self.description,
            "domains": [
                "AtomicPhysics", "ClassicalElectromagnetism", "ClassicalMechanics",
                "Electrodynamics", "GeometricalOptics", "QuantumMechanics",
                "Relativity", "SemiconductorPhysics", "Solid-StatePhysics",
                "StatisticalMechanics", "TheoreticalMechanics", "Thermodynamics", "WaveOptics"
            ],
            "languages": ["en", "zh"],
            "variants": ["mini", "full"],
            "splits": ["test"],  # UGPhysics only supports test split
            "problem_types": ["OE"],  # All problems are open-ended
            "total_problems": "~2000+",
            "difficulty": "undergraduate",
            "source": "UGPhysics Dataset"
        }
    
    @property
    def field_mapping(self) -> Dict[str, str]:
        """
        Define field mapping from UGPhysics fields to standard PhysKit fields.
        
        Returns:
            Dictionary mapping UGPhysics field names to standard field names
        """
        return {
            "index": "problem_id",      # Map "index" to "problem_id"
            "problem": "question",      # Map "problem" to "question"
            "solution": "solution",     # Map "solution" to "solution"
            "domain": "domain",         # Map "domain" to "domain"
            "topic": "topic",           # Map "topic" to topic metadata
            "language": "language",     # Map "language" to language metadata
        }
    
    def _process_metadata(self, metadata: Dict[str, Any], domain_name: str):
        """Process metadata to create standardized problem fields."""
        # Set domain from the domain name
        metadata['domain'] = domain_name
        
        # difficulty
        metadata['difficulty'] = "undergraduate"


        # answer type and answer        
        answer_type = metadata.get('answer_type')
        if "NV" in answer_type:
            metadata['answer_type'] = 'numerical'
            answer_value = metadata.get('answers')
            answer_unit = metadata.get('unit')
            metadata['answer'] = {
                "value": answer_value,
                "unit": answer_unit
            }
        elif "EX" in answer_type:
            metadata['answer_type'] = 'symbolic'
            metadata['answer'] = metadata.get('answers')
        else:
            metadata['answer_type'] = 'textual'
            metadata['answer'] = metadata.get('answers')
        
        metadata.pop('answers', None)
        metadata.pop('unit', None)
        
        return metadata
    
    def load(
        self,
        data_dir: Union[str, Path, None] = None,
        split: str = "test",
        variant: str = "full",
        sample_size: Optional[int] = None,
        per_domain: Optional[int] = None,
        language: str = "en",
        **kwargs
    ) -> PhysicalDataset:
        """
        Load the UGPhysics dataset.
        
        Args:
            data_dir: Path to the UGPhysics dataset (defaults to ~/data/ugphysics)
            split: Dataset split to load (only "test" is supported)
            variant: Dataset variant ("full" only)
            sample_size: Number of problems to sample (None for all)
            per_domain: Number of problems to sample per domain (None for all)
            language: Language to load ("en" or "zh")
            
        Returns:
            PhysicalDataset instance
            
        Raises:
            ValueError: If unsupported split is requested
        """
        if split != "test":
            raise ValueError("UGPhysics dataset only supports 'test' split")
        
        # Resolve data directory with environment variable support
        data_dir = self.resolve_data_dir(data_dir, "ugphysics")
        print(f"🔍 Using data directory: {data_dir}")
        
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        if variant != "full":
            raise ValueError(f"Unsupported variant: {variant}. Supported variants: 'full'")
        
        # Get available domains
        domains = self._get_domains(data_dir, language)
        
        if not domains:
            print(f"❌ No valid domains found in: {data_dir}")
            print(f"📁 Directory contents: {[d.name for d in data_dir.iterdir() if d.is_dir()]}")
            print(f"💡 Expected structure:")
            print(f"   {data_dir}/")
            print(f"   ├── AtomicPhysics/")
            print(f"   │   └── en.jsonl")
            print(f"   ├── ClassicalMechanics/")
            print(f"   │   └── en.jsonl")
            print(f"   └── ... (other domains)")
            return PhysicalDataset([], self.get_info(), split=split)
        
        problems = []
        domain_counts = {}
        domain_problems = {}  # Track problems per domain for sampling

        for domain_name in domains:
            domain_file = data_dir / domain_name / f"{language}.jsonl"
            
            if not domain_file.exists():
                print(f"Domain file not found: {domain_file}")
                continue
            
            domain_problems[domain_name] = []  # Initialize domain problems list
            
            try:
                with open(domain_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        if line.strip():
                            try:
                                data = json.loads(line)
                                
                                metadata = self.intiailize_metadata(data)
                                metadata = self._process_metadata(metadata, domain_name)
                                
                                # Create the physics problem using base class method
                                problem = self.create_physics_problem(
                                    metadata=metadata,
                                )
                                
                                domain_problems[domain_name].append(problem)
                                
                            except json.JSONDecodeError as e:
                                print(f"Error parsing JSON in {domain_file}:{line_num}: {e}")
                                continue
                            except Exception as e:
                                print(f"Error loading problem from {domain_file}:{line_num}: {e}")
                                continue
            except Exception as e:
                print(f"Error loading domain {domain_name}: {e}")
                continue
        
        # Apply per_domain sampling if requested
        if per_domain is not None:
            for domain_name, domain_problem_list in domain_problems.items():
                if len(domain_problem_list) > per_domain:
                    domain_problems[domain_name] = random.sample(domain_problem_list, per_domain)
        
        # Collect all problems and count by domain
        for domain_name, domain_problem_list in domain_problems.items():
            problems.extend(domain_problem_list)
            domain_counts[domain_name] = len(domain_problem_list)
        
        # Apply overall sample_size sampling if requested
        if sample_size is not None and sample_size < len(problems):
            problems = random.sample(problems, sample_size)
        
        # Create dataset info
        info = self.get_info()
        info["total_problems"] = len(problems)
        info["problems_by_domain"] = {str(domain): count for domain, count in domain_counts.items()}
        
        return PhysicalDataset(
            problems,
            info,
            split=split,
        )
    
    def _get_domains(
        self,
        data_dir: Union[str, Path],
        language: str = "en"
    ) -> List[str]:
        """Get list of available domains in the dataset."""
        data_dir = Path(data_dir)
        if not data_dir.exists():
            print(f"❌ Data directory does not exist: {data_dir}")
            return []
        
        print(f"🔍 Scanning directory: {data_dir}")
        
        # Get all subdirectories
        all_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
        
        # Get subdirectories that represent domains
        domains = [d.name for d in all_dirs]
        
        # Filter out non-domain directories
        valid_domains = [
            "AtomicPhysics", "ClassicalElectromagnetism", "ClassicalMechanics",
            "Electrodynamics", "GeometricalOptics", "QuantumMechanics",
            "Relativity", "SemiconductorPhysics", "Solid-StatePhysics",
            "StatisticalMechanics", "TheoreticalMechanics", "Thermodynamics", "WaveOptics"
        ]
        
        found_domains = [d for d in domains if d in valid_domains]
        
        # Check if any domain files exist
        for domain in found_domains:
            domain_dir = data_dir / domain
            jsonl_file = domain_dir / "en.jsonl"
            if not jsonl_file.exists():
                print(f"⚠️  Domain {domain}: {language}.jsonl not found")
        
        return found_domains
    