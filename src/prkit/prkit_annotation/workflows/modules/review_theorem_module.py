"""
Theorem review workflow module for physics problems.

This module provides theorem review functionality that allows human reviewers
to evaluate predicted theorems for relevance, equation correctness, and condition validity.
It can be composed into larger annotation workflows.
"""

from typing import Any, Dict, Optional, List

from prkit.prkit_core.models.physics_problem import PhysicsProblem
from datetime import datetime
from prkit.prkit_annotation.workers import TheoremDetector
from .base_module import BaseWorkflowModule


class ReviewTheoremModule(BaseWorkflowModule):
    """
    Workflow module for theorem review in physics problems.
    
    This module allows human reviewers to evaluate predicted theorems for:
    1. Relevance to the physics problem
    2. Correctness of equations
    3. Validity of conditions
    
    It processes theorems that have been previously detected by other modules
    and provides a structured review interface for human evaluation.
    """
    
    def __init__(
        self,
        name: str = "theorem_reviewer",
        model: str = "o3-mini",
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name, model, config)
        
        # Initialize the theorem detector
        self.theorem_detector = TheoremDetector(model=model)
        
        # Override module status with theorem review-specific information
        self.module_status.update({
            "review_type": "theorem",
            "theorems_reviewed": 0,
            "problems_with_relevant_theorems": 0,
            "problems_without_relevant_theorems": 0,
            "average_theorems_per_problem": 0.0,
            "relevant_theorems_count": 0,
            "correct_equations_count": 0,
            "valid_conditions_count": 0,
            # Missing theorem statistics
            "missing_theorems_added": 0,
            "problems_with_missing_theorems": 0,
            # Ensure generic fields are present
            "total_problems": 0,
            "successful_problems": 0,
            "failed_problems": 0
        })
    
    def process(
        self,
        problem: PhysicsProblem,
        **kwargs
    ) -> dict[str, Any]:
        """
        Process input data and return theorem review results.
        
        The output is a dictionary with the following keys:
        - theorems: List of reviewed theorems
        - missing_theorems: List of missing theorems
        - review_metadata: Dictionary containing review statistics
        
        Args:
            problem: PhysicsProblem object containing the problem and predicted theorems
            **kwargs: Additional arguments
            
        Returns:
            Theorem review result
        """
        # Import here to avoid circular imports
        
        # Extract question text, and theorems from various input formats
        question = problem.question
        solution = problem.solution if problem.solution else ""
        problem_id = problem.problem_id
        theorems = problem.additional_fields.get("theorems", [])
        
        self.logger.info(
            "Starting theorem review for problem %s with %d theorems",
            problem_id,
            len(theorems),
        )
        
        review_stats = {
            "reviewed_theorems": 0,
            "relevant_theorems": 0,
            "correct_equations": 0,
            "valid_conditions": 0,
            "missing_theorems": 0,
            "total_theorems": len(theorems)
        }
        
        # Review the predicted theorems
        if theorems:
            reviewed_theorems = []
            
            for i, theorem in enumerate(theorems):
                self.logger.info("Reviewing theorem %d/%d: %s", i+1, len(theorems), theorem.get('name', 'Unknown'))
                
                reviewed_theorem = self._review_single_theorem(
                    theorem=theorem,
                    problem_question=question,
                    problem_solution=solution,
                    theorem_index=i+1,
                    total_theorems=len(theorems)
                )
                
                reviewed_theorems.append(reviewed_theorem)
                review_stats["reviewed_theorems"] += 1
                
                # Update statistics
                if reviewed_theorem.get("is_relevant", False):
                    review_stats["relevant_theorems"] += 1
                if reviewed_theorem.get("equations_correct", False):
                    review_stats["correct_equations"] += 1
                if reviewed_theorem.get("conditions_valid", False):
                    review_stats["valid_conditions"] += 1
            
            # Update module status
            self.module_status["total_problems"] += 1
            self.module_status["successful_problems"] += 1
            self.module_status["theorems_reviewed"] += len(theorems)
            self.module_status["relevant_theorems_count"] += review_stats["relevant_theorems"]
            self.module_status["correct_equations_count"] += review_stats["correct_equations"]
            self.module_status["valid_conditions_count"] += review_stats["valid_conditions"]
            
            if review_stats["relevant_theorems"] > 0:
                self.module_status["problems_with_relevant_theorems"] += 1
            else:
                self.module_status["problems_without_relevant_theorems"] += 1
            
            # Calculate average theorems per problem
            if self.module_status["total_problems"] > 0:
                self.module_status["average_theorems_per_problem"] = (
                    self.module_status["theorems_reviewed"] / self.module_status["total_problems"]
                )
        
        # Review missing theorems
        missing_theorems = self._review_missing_theorems(problem)
        
        result = {
            "theorems": reviewed_theorems,
            "missing_theorems": missing_theorems,
            "review_metadata": {
                "problem_id": problem_id,
                **review_stats,
                "missing_theorems_count": len(missing_theorems)
            }
        }
        
        self.logger.info("Completed theorem review for problem %s: %d/%d relevant", problem_id, review_stats['relevant_theorems'], review_stats['total_theorems'])
        
        return result
    
    def _review_missing_theorems(self, problem: PhysicsProblem):
        """
        Review missing theorems for a problem.
        
        This function allows the reviewer to input theorem names and fill in the details
        for theorems that were not detected by the automated system.
        
        Args:
            problem: PhysicsProblem object containing the problem and predicted theorems
        """
        question = problem.question
        solution = problem.solution if problem.solution else ""
        problem_id = problem.problem_id
        
        self.logger.info("Starting missing theorem review for problem %s", problem_id)
        
        print("\n" + "="*80)
        print("MISSING THEOREM REVIEW")
        print("="*80)
        print("PROBLEM:")
        print("-" * 40)
        print(question)
        print("-" * 40)
        
        if solution:
            print("\nSOLUTION:")
            print("-" * 40)
            print(solution)
            print("-" * 40)
        
        print("\nDo you think there are any important theorems missing from the detected ones?")
        print("You can add missing theorems by entering their names.")
        print("Enter 'DONE' when you're finished adding missing theorems.")
        print("="*80)
        
        missing_theorems = []
        theorem_counter = 1
        
        while True:
            print(f"\n--- Missing Theorem {theorem_counter} ---")
            theorem_name = input("Enter theorem name (or 'DONE' to finish): ").strip()
            
            if theorem_name.upper() == 'DONE' or theorem_name == '':
                break
            
            if theorem_name:
                # Create a new theorem entry
                missing_theorem = {
                    "name": theorem_name,
                    "description": "",
                    "equations": [],
                    "domain": "",
                    "conditions": [],
                    "is_missing_theorem": True,  # Flag to indicate this was added manually
                    "review_timestamp": datetime.now().isoformat()
                }
                
                # Get description
                description = input(f"Enter description for '{theorem_name}': ").strip()
                if description:
                    missing_theorem["description"] = description
                
                # Get equations
                print(f"Enter equations for '{theorem_name}' (one per line, empty line to finish):")
                equations = []
                while True:
                    equation = input("  Equation: ").strip()
                    if equation == '':
                        break
                    equations.append(equation)
                missing_theorem["equations"] = equations
                
                # Get domain
                domain = input(f"Enter domain for '{theorem_name}': ").strip()
                if domain:
                    missing_theorem["domain"] = domain
                
                # Get conditions
                print(f"Enter conditions for '{theorem_name}' (one per line, empty line to finish):")
                conditions = []
                while True:
                    condition = input("  Condition: ").strip()
                    if condition == '':
                        break
                    conditions.append(condition)
                missing_theorem["conditions"] = conditions
                
                # Mark missing theorem as relevant and correct (since human added it)
                missing_theorem.update({
                    "is_relevant": True,
                    "equations_correct": True,
                    "conditions_valid": True,
                    "relevance_feedback": "Added by human reviewer - assumed relevant",
                    "equations_feedback": "Added by human reviewer - assumed correct",
                    "conditions_feedback": "Added by human reviewer - assumed valid"
                })
                
                missing_theorems.append(missing_theorem)
                theorem_counter += 1
                
                print(f"✓ Added missing theorem: {theorem_name}")
        
        # Update module status with missing theorem statistics
        if missing_theorems:
            self.module_status["missing_theorems_added"] = len(missing_theorems)
            self.module_status["problems_with_missing_theorems"] = 1
            
            self.logger.info("Added %d missing theorems for problem %s", 
                           len(missing_theorems), problem_id)
        else:
            self.module_status["missing_theorems_added"] = 0
            self.module_status["problems_with_missing_theorems"] = 0
            self.logger.info("No missing theorems added for problem %s", problem_id)
        
        return missing_theorems
    
    def _review_single_theorem(
        self,
        theorem: Dict[str, Any],
        problem_question: str,
        problem_solution: str,
        theorem_index: int,
        total_theorems: int
    ) -> Dict[str, Any]:
        """
        Review a single theorem with human feedback.
        
        Args:
            theorem: The theorem dictionary to review
            problem_question: The physics problem question text
            problem_solution: The physics problem solution text
            theorem_index: Index of this theorem (1-based)
            total_theorems: Total number of theorems to review
            
        Returns:
            Dictionary containing the original theorem plus review feedback
        """
        theorem_name = theorem.get("name", "Unknown Theorem")
        
        self.logger.info("Starting human review for theorem: %s", theorem_name)
        
        # Display theorem information for human review
        print("\n" + "="*80)
        print(f"THEOREM REVIEW - {theorem_index}/{total_theorems}")
        print("="*80)
        print("PROBLEM:")
        print("-" * 40)
        print(problem_question)
        print("-" * 40)
        
        if problem_solution:
            print("\nSOLUTION:")
            print("-" * 40)
            print(problem_solution)
            print("-" * 40)
        
        print(f"\nTheorem: {theorem_name}")
        print(f"Description: {theorem.get('description', 'No description provided')}")
        
        if theorem.get('equations'):
            print("\nEquations:")
            for eq in theorem['equations']:
                print(f"  - {eq}")
        
        if theorem.get('conditions'):
            print("\nConditions:")
            for i, condition in enumerate(theorem['conditions'], 1):
                print(f"  {i}. {condition}")
        
        print("\n" + "="*80)
        
        # Step 1: Check relevance
        print("\n1. RELEVANCE CHECK")
        print("Is this theorem relevant to solving the given physics problem?")
        is_relevant = self._get_human_feedback(
            prompt="Enter 'y' for yes, 'n' for no: ",
            valid_responses=['y', 'n', 'yes', 'no']
        )
        
        relevance_feedback = ""
        if is_relevant.lower() in ['y', 'yes']:
            print("✓ Theorem is relevant to the problem")
            relevance_feedback = "Theorem is relevant to solving the physics problem"
            
            # Step 2: Check equations (only if relevant)
            print("\n2. EQUATIONS CHECK")
            print("Are the equations correct and appropriate for this theorem?")
            equations_correct = self._get_human_feedback(
                prompt="Enter 'y' for yes, 'n' for no: ",
                valid_responses=['y', 'n', 'yes', 'no']
            )
            
            equations_feedback = ""
            if equations_correct.lower() in ['y', 'yes']:
                print("✓ Equations are correct")
                equations_feedback = "Equations are correct and appropriate for this theorem"
            else:
                print("✗ Equations need correction")
                equations_feedback = input("Please provide feedback on what's wrong with the equations: ").strip()
            
            # Step 3: Check conditions (only if relevant and equations are correct)
            print("\n3. CONDITIONS CHECK")
            print("Do the conditions make sense and are they appropriate for this theorem?")
            conditions_valid = self._get_human_feedback(
                prompt="Enter 'y' for yes, 'n' for no: ",
                valid_responses=['y', 'n', 'yes', 'no']
            )
            
            conditions_feedback = ""
            if conditions_valid.lower() in ['y', 'yes']:
                print("✓ Conditions are valid")
                conditions_feedback = "Conditions are appropriate and make sense for this theorem"
            else:
                print("✗ Conditions need improvement")
                conditions_feedback = input("Please provide feedback on what's wrong with the conditions: ").strip()
        else:
            print("✗ Theorem is not relevant to the problem")
            relevance_feedback = input("Please provide feedback on why this theorem is not relevant: ").strip()
            equations_correct = 'n'
            conditions_valid = 'n'
            equations_feedback = "Not applicable - theorem is not relevant"
            conditions_feedback = "Not applicable - theorem is not relevant"
        
        # Create reviewed theorem with feedback
        reviewed_theorem = theorem.copy()
        reviewed_theorem.update({
            "is_relevant": is_relevant.lower() in ['y', 'yes'],
            "equations_correct": equations_correct.lower() in ['y', 'yes'],
            "conditions_valid": conditions_valid.lower() in ['y', 'yes'],
            "relevance_feedback": relevance_feedback,
            "equations_feedback": equations_feedback,
            "conditions_feedback": conditions_feedback,
            "review_timestamp": datetime.now().isoformat(),
            "theorem_index": theorem_index
        })
        
        self.logger.info("Completed review for theorem %s: relevant=%s, equations_correct=%s, conditions_valid=%s", theorem_name, reviewed_theorem['is_relevant'], reviewed_theorem['equations_correct'], reviewed_theorem['conditions_valid'])
        
        return reviewed_theorem
    
    def _get_human_feedback(
        self,
        prompt: str,
        valid_responses: List[str]
    ) -> str:
        """
        Get human feedback with input validation.
        
        Args:
            prompt: The prompt to display to the user
            valid_responses: List of valid response options
            
        Returns:
            The user's response (validated)
        """
        while True:
            try:
                response = input(prompt).strip().lower()
                if response in valid_responses:
                    return response
                else:
                    print(f"Invalid response. Please enter one of: {', '.join(valid_responses)}")
            except KeyboardInterrupt as exc:
                print("\n\nReview interrupted by user. Exiting...")
                raise KeyboardInterrupt("Review interrupted by user") from exc
            except (EOFError, OSError) as e:
                print(f"Error getting input: {e}")
                continue
    
    def reset(self) -> None:
        """Reset module state and status with all required fields."""
        super().reset()  # Call parent reset first
        
        # Ensure all theorem review-specific fields are present
        self.module_status.update({
            "review_type": "theorem",
            "theorems_reviewed": 0,
            "problems_with_relevant_theorems": 0,
            "problems_without_relevant_theorems": 0,
            "average_theorems_per_problem": 0.0,
            "relevant_theorems_count": 0,
            "correct_equations_count": 0,
            "valid_conditions_count": 0,
            # Missing theorem statistics
            "missing_theorems_added": 0,
            "problems_with_missing_theorems": 0
        })
    
    def get_status(self) -> Dict[str, Any]:
        """Get current module status with safety checks."""
        # Ensure all generic fields exist before returning status
        self.module_status.setdefault("total_problems", 0)
        self.module_status.setdefault("successful_problems", 0)
        self.module_status.setdefault("failed_problems", 0)
        # Ensure all theorem review-specific fields exist
        self.module_status.setdefault("theorems_reviewed", 0)
        self.module_status.setdefault("problems_with_relevant_theorems", 0)
        self.module_status.setdefault("problems_without_relevant_theorems", 0)
        self.module_status.setdefault("average_theorems_per_problem", 0.0)
        self.module_status.setdefault("relevant_theorems_count", 0)
        self.module_status.setdefault("correct_equations_count", 0)
        self.module_status.setdefault("valid_conditions_count", 0)
        # Missing theorem statistics
        self.module_status.setdefault("missing_theorems_added", 0)
        self.module_status.setdefault("problems_with_missing_theorems", 0)
        return super().get_status()
    
    def _form_output_as_a_problem(
        self,
        result: dict[str, Any],
        problem: PhysicsProblem
    ) -> PhysicsProblem:
        """
        Form the output as a PhysicsProblem object.
        
        Args:
            result: The theorem review result dictionary containing:
                - theorems: List of reviewed theorems
                - missing_theorems: List of missing theorems  
                - review_metadata: Dictionary containing review statistics
            problem: The original PhysicsProblem object
            
        Returns:
            PhysicsProblem object with theorem review results properly structured
        """
        
        # Debug: log what we're receiving
        self.logger.info("_form_output_as_a_problem called with result type: %s", type(result))
        self.logger.info("Result content: %s", result)
        
        # Create a copy of the original problem
        new_problem = problem.copy()
        
        # Ensure additional_fields exists
        if not hasattr(new_problem, 'additional_fields') or new_problem.additional_fields is None:
            new_problem.additional_fields = {}
        
        # Remove "theorems" from additional_fields (if it exists)
        new_problem.additional_fields.pop("theorems", None)
        
        theorem_metadata = {}
        detection_metadata = new_problem.additional_fields.pop(
            "theorem_detection_metadata",
            None
        )
        if detection_metadata is not None:
            theorem_metadata["detection"] = detection_metadata
        
        if "review_metadata" in result:
            theorem_metadata["review"] = result["review_metadata"]
        
        if theorem_metadata:
            new_problem.additional_fields["theorem_metadata"] = theorem_metadata
        
        # Add the reviewed and missing theorems
        new_problem.additional_fields["reviewed_theorems"] = result.get("theorems", [])
        new_problem.additional_fields["missing_theorems"] = result.get("missing_theorems", [])
        
        return new_problem
