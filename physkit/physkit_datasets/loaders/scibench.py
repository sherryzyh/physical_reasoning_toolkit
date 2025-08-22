"""
SciBench Dataset Loader

This loader supports the SciBench dataset which contains college-level scientific problems
from textbooks across Mathematics, Chemistry, and Physics. The dataset includes:

- Subjects: Mathematics (calculus, diff, fund, stat), Chemistry (atkins, chemmc, quan), Physics (class, matter, thermo)
- Question Types: Open-ended problems with step-by-step solutions
- Format: JSON with LaTeX-formatted questions, answers, and detailed solutions
- Source: College textbook problems and solutions

The loader automatically determines problem types, maps SciBench fields to standard PhysKit fields,
and merges problems with their corresponding solutions using problemid as the identifier.

"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from .base_loader import BaseDatasetLoader
from physkit_core.models import PhysicalDataset, PhysicsProblem
from physkit_core.definitions.physics_domain import PhysicsDomain
from physkit_core.definitions.answer_types import AnswerType
from physkit_core import PhysKitLogger


class SciBenchLoader(BaseDatasetLoader):
    """Loader for SciBench dataset with support for subject filtering and solution merging."""

    def __init__(self):
        """Initialize the SciBench loader with a logger."""
        super().__init__()
        self.logger = PhysKitLogger.get_logger(__name__)
    
    @property
    def DOMAIN_SUBJECT_MAPPING(self) -> Dict[str, str]:
        """Return mapping of source domains to their subjects and problem counts."""
        # Subject classification mapping
        return {
            "diff": {
            "domain": "Differential Equations",
            "subject": "math",
            "problem_count": 50
            },
            "fund": {
            "domain": "Fundamental Physics",
            "subject": "physics",
            "problem_count": 71
            },
            "chemmc": {
            "domain": "Chemistry Multiple Choice",
            "subject": "chemistry",
            "problem_count": 38
            },
            "atkins": {
            "domain": "Physical Chemistry",
            "subject": "chemistry",
            "problem_count": 105
            },
            "matter": {
            "domain": "Matter and Materials",
            "subject": "chemistry",
            "problem_count": 47
            },
            "thermo": {
            "domain": "Thermodynamics",
            "subject": "chemistry",
            "problem_count": 66
            },
            "class": {
            "domain": "Classical Mechanics",
            "subject": "physics",
            "problem_count": 56
            },
            "quan": {
            "domain": "Quantum Mechanics",
            "subject": "physics",
            "problem_count": 33
            },
            "stat": {
            "domain": "Statistics",
            "subject": "math",
            "problem_count": 72
            },
            "calculus": {
            "domain": "Calculus",
            "subject": "math",
            "problem_count": 42
            }
        }

    @property
    def name(self) -> str:
        return "scibench"

    @property
    def description(self) -> str:
        return (
            "SciBench: College-level scientific problems from textbooks across Mathematics, "
            "Chemistry, and Physics with detailed step-by-step solutions"
        )

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "domains": [
                "fundamental_physics",
                "classical_mechanics",
                "quantum_mechanics"
            ],
            "languages": ["en"],
            "variants": ["full"],
            "splits": ["test"],
            "problem_types": ["OE"],
            "total_problems": "160",
            "difficulty": "College level",
            "source": "College textbooks",
            "citation": "SciBench: A College-Level Scientific Reasoning Benchmark",
        }

    @property
    def field_mapping(self) -> Dict[str, str]:
        """
        Define field mapping from SciBench fields to standard PhysKit fields.
        
        SciBench fields:
        - problem_text: question text
        - problemid: problem identifier
        - answer_latex: LaTeX formatted answer
        - answer_number: numerical answer
        - unit: unit of measurement
        - source: subject source (calculus, atkins, class, etc.)
        - comment: additional comments
        - solution: detailed solution steps
        """
        return {
            "problem_text": "question",
            "solution": "solution",
            "domain": "domain",
        }


    def _parse_unit_with_exponent(self, unit: str) -> Tuple[str, str]:
        """
        Parse unit string to extract exponent and actual unit.
        
        Examples:
        - "10^4 m/s" -> ("4", "m/s")
        - " 10^-3 kg" -> ("-3", "kg")
        - "10^5" -> ("5", "")
        - "m/s" -> ("", "m/s")
        - "" -> ("", "")
        
        Args:
            unit: Unit string that may contain exponent and/or unit
            
        Returns:
            Tuple of (exponent, actual_unit)
        """
        unit = unit.strip()
        
        if not unit:
            return "", ""
        
        # Check if unit starts with scientific notation
        if unit.startswith("10^"):
            # Find where the exponent ends
            parts = unit.split(" ", 1)  # Split on first space
            if len(parts) == 2:
                # Case: "10^4 m/s"
                exponent_part = parts[0]  # "10^4"
                actual_unit = parts[1].strip()  # "m/s"
                exponent = exponent_part.replace("10^", "").strip()
                return exponent, actual_unit
            else:
                # Case: "10^4" (no unit after exponent)
                exponent = unit.replace("10^", "").strip()
                return exponent, ""
        else:
            # Case: "m/s" (no exponent, just unit)
            return "", unit

    def _determine_answer_and_type(
        self, 
        answer_latex: str, 
        answer_number: str, 
        unit: str
    ) -> Tuple[str, str, str]:
        
        # Parse unit to extract exponent and actual unit
        exponent, actual_unit = self._parse_unit_with_exponent(unit)
        
        # Handle scientific notation in unit (e.g., "10^5 m/s" -> "111.1e5", "m/s")
        if exponent:
            try:
                # Convert to proper scientific notation format
                if answer_number and answer_number.strip():
                    # Convert "111.1x10^5" format to "111.1e5" format
                    answer_value = f"{answer_number}e{exponent}"
                    return answer_value, "numerical", actual_unit
                else:
                    # If no answer_number, use answer_latex
                    if answer_latex and answer_latex.strip():
                        answer_value = f"{answer_latex}e{exponent}"
                        return answer_value, "numerical", actual_unit
                    else:
                        # If no answer data but we have a unit, return empty with numerical type
                        return "", "numerical", actual_unit
            except Exception:
                # Fallback to original format if conversion fails
                answer_value = f"{answer_number}x{unit}" if answer_number else f"{answer_latex}x{unit}"
                return answer_value, "numerical", ""

        # Handle scientific notation in answer_latex (e.g., "111.1x10^5")
        if answer_latex and "x10^" in answer_latex:
            try:
                # Convert "111.1x10^5" to "111.1e5"
                answer_value = answer_latex.replace("x10^", "e")
                return answer_value, "numerical", actual_unit
            except Exception:
                pass

        # Numerical answer (when answer_latex and answer_number are the same)
        if answer_number and answer_number.strip() != "" and answer_latex and answer_latex.strip() != "" and answer_latex.strip() == answer_number.strip():
            return answer_number, "numerical", actual_unit
        
        # Symbolic answer (if answer_latex contains mathematical expressions)
        if answer_latex and answer_latex.strip() != "":
            # Check if it's symbolic (contains mathematical symbols)
            if any(symbol in answer_latex for symbol in ['=', '+', '-', '*', '/', '^', '\\', '{', '}']):
                return answer_latex, "symbolic", actual_unit
            else:
                # If no mathematical symbols, treat as numerical
                return answer_latex, "numerical", actual_unit

        # Fallback: try to determine if answer_number is numerical
        if answer_number and answer_number.strip() != "":
            try:
                float(answer_number)
                return answer_number, "numerical", actual_unit
            except (ValueError, TypeError):
                pass
        
        # Default to textual if nothing else matches
        return answer_latex or answer_number or "", "textual", actual_unit

    def _load_json_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load and parse a JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError(f"Expected data to be a list, got {type(data)}: {data}")
                for item in data:
                    if not isinstance(item, dict):
                        raise ValueError(f"Expected item to be a dict, got {type(item)}: {item}")
                return data
        except Exception as e:
            self.logger.warning(f"Could not load {file_path}: {e}")
            return []

    def _merge_problems_with_solutions(
        self, 
        problems: List[Dict[str, Any]], 
        solutions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merge problems with their corresponding solutions using problemid as identifier.
        
        Args:
            problems: List of problem dictionaries
            solutions: List of solution dictionaries
            
        Returns:
            List of merged problem dictionaries with solutions
        """
        # Create solution lookup
        solution_lookup = {}
        for solution in solutions:
            problemid = solution.get("problemid", "").strip()
            if problemid:
                solution_lookup[problemid] = solution
        
        # Merge problems with solutions
        merged_problems = []
        for problem in problems:
            problemid = problem.get("problemid", "").strip()
            
            # Create merged problem
            merged_problem = problem.copy()
            
            # Add solution if available
            if problemid in solution_lookup:
                solution_data = solution_lookup[problemid]
                merged_problem["solution"] = solution_data.get("solution", "")
                # Update other fields if they're more complete in solution file
                for field in ["answer_latex", "answer_number", "unit", "comment"]:
                    if field in solution_data and solution_data[field]:
                        merged_problem[field] = solution_data[field]
            else:
                # Keep empty solution if no match found
                merged_problem["solution"] = ""
            
            merged_problems.append(merged_problem)
        
        return merged_problems

    @property
    def DOMAIN_MAPPING(self) -> Dict[str, str]:
        """Mapping of domain abbreviations to full domain names."""
        return {
            "fund": PhysicsDomain.FUNDAMENTAL_PHYSICS,
            "class": PhysicsDomain.CLASSICAL_MECHANICS,
            "quan": PhysicsDomain.QUANTUM_MECHANICS,
        }

    def _process_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process metadata to create a PhysicsProblem."""
        
        # Determine answer type and extract unit
        answer_latex = metadata.get("answer_latex", "")
        answer_number = metadata.get("answer_number", "")
        unit = metadata.get("unit", "")
        answer_value, answer_type, extracted_unit = self._determine_answer_and_type(
            answer_latex,
            answer_number,
            unit
        )
        
        # Remove answer_latex, answer_number, and unit
        metadata.pop("answer_latex", None)
        metadata.pop("answer_number", None)
        metadata.pop("unit", None)
        
        # Set the processed values
        metadata['answer'] = answer_value
        metadata['answer_type'] = answer_type
        if answer_type == "numerical" and extracted_unit:
            metadata['answer'] = {
                "value": answer_value,
                "unit": extracted_unit
            }
        else:
            metadata['answer'] = answer_value
        
        # Set problem type
        metadata['problem_type'] = "OE"
        
        domain = metadata.get("domain", "")
        if domain:
            metadata['domain'] = self.DOMAIN_MAPPING.get(domain, PhysicsDomain.OTHER)
        
        problem_id = metadata.get("problemid", "")
        metadata.pop("problemid", None)
        metadata["problem_id"] = domain + "_" + problem_id.strip()
        return metadata


    def load(
        self,
        data_dir: Union[str, Path, None] = None,
        sample_size: Optional[int] = None,
        **kwargs
    ) -> PhysicalDataset:
        """
        Load SciBench dataset from the specified directory.
        
        Args:
            data_dir: Directory containing the SciBench dataset
            **kwargs: Additional loading parameters
            
        Returns:
            PhysicalDataset instance
        """
        data_dir = self.resolve_data_dir(data_dir, "scibench")
        
        all_problems = []
        
        # Get all JSON files in the directory
        json_files = list(data_dir.glob("*.json"))
        
        # Separate problem files and solution files
        problem_files = [f for f in json_files if not f.name.endswith("_sol.json")]
        solution_files = [f for f in json_files if f.name.endswith("_sol.json")]
        
        self.logger.debug("Found %d problem files and %d solution files", len(problem_files), len(solution_files))
        
        
        # Process each problem file
        for problem_file in problem_files:
            source_name = problem_file.stem  # filename without extension
            if source_name not in self.DOMAIN_SUBJECT_MAPPING:
                continue
            
            # Load problems
            problems = self._load_json_file(problem_file)
            if not problems:
                continue
            
            # Find corresponding solution file
            solution_file = problem_file.parent / f"{source_name}_sol.json"
            solutions = []
            if solution_file.exists():
                solutions = self._load_json_file(solution_file)
            
            # Merge problems with solutions
            merged_problems = self._merge_problems_with_solutions(problems, solutions)
            
            # Filter by subject and add to all problems
            for problem in merged_problems:
                problem["domain"] = source_name
            
            all_problems.extend(merged_problems)
        
        self.logger.debug("Loaded %d problems", len(all_problems))
        
        # Create PhysicsProblem objects
        physics_problems = []
        for idx, problem_data in enumerate(all_problems):
            try:
                metadata = self.initialize_metadata(problem_data)
                metadata = self._process_metadata(metadata)
                physics_problem = self.create_physics_problem(metadata)
                physics_problems.append(physics_problem)
            except Exception as e:
                self.logger.warning("Could not create problem from %s: %s", problem_data.get('problemid', 'unknown'), e)
                continue
        
        self.logger.debug("Successfully created %d PhysicsProblem objects", len(physics_problems))
        
        # Create PhysicalDataset
        dataset = PhysicalDataset(
            problems=physics_problems,
            info={
                "name": self.name,
                "description": self.description,
                "total_problems": len(physics_problems),
                "subjects_included": list(set(p.domain.value for p in physics_problems if p.domain)),
                "source_files": [f.stem for f in problem_files]
            }
        )
        
        # Log final loading result
        self.logger.info(f"Successfully loaded {len(physics_problems)} problems from SciBench dataset")
        
        return dataset
