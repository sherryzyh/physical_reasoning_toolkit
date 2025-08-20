from dataclasses import dataclass

@dataclass
class PhysicsProblemMetadata:
    """
    Metadata for a physics problem.
    """
    problem_id: str
    question: str
    answer: str
    solution: str
    domain: str