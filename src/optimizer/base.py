
from typing import Dict, Any, Optional

from dspy.teleprompt import LabeledFewShot, GEPA, BootstrapFewShot
from src.metrics import (
    strict_view_selector_metric, 
    soft_view_selector_metric,
    calculate_metrics,
    print_evaluation_report
)

class OptimizerFactory:
    """Factory for creating DSPy optimizers."""
    
    @staticmethod
    def create_labeledfewshot(config: Dict[str, Any]):
        """Create LabeledFewShot optimizer."""
        return LabeledFewShot(k=config.get('k', 10))
    
    @staticmethod
    def create_gepa(config: Dict[str, Any], reflection_lm):
        """Create GEPA optimizer."""
        return GEPA(
            metric=soft_view_selector_metric,
            reflection_lm=reflection_lm,
            num_threads=config.get('num_threads', 1),
            max_full_evals=config.get('max_full_evals', 5)
        )
    
    @staticmethod
    def create_bootstrap(config: Dict[str, Any]):
        """Create BootstrapFewShot optimizer."""
        return BootstrapFewShot(
            metric=strict_view_selector_metric,
            max_bootstrapped_demos=config.get('max_bootstrapped_demos', 8),
            max_labeled_demos=config.get('max_labeled_demos', 8),
            max_rounds=config.get('max_rounds', 5),
            max_errors=config.get('max_errors', 1)
        )