"""
Evaluation runner for baseline and optimized modules.
"""

import dspy
from typing import List, Dict, Any
from tqdm import tqdm
from .modules import ViewSelectorModule
from .data_loader import parse_view_list
from .metrics import calculate_precision, calculate_recall, calculate_f1, calculate_accuracy


class ViewSelectorEvaluator:
    """Evaluator for ViewSelector modules."""
    
    def __init__(self, module: ViewSelectorModule):
        self.module = module
    
    def evaluate_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single example.
        
        Args:
            example: Dictionary with 'question', 'expected_views', etc.
            
        Returns:
            Dictionary with prediction results and metrics
        """
        question = example['question']
        conversation_history = example.get('conversation_history', '')
        expected_views_str = example['expected_views']
        
        # Get prediction
        try:
            prediction = self.module(
                question=question,
                conversation_history=conversation_history
            )
            
            predicted_views_str = prediction.selected_views
            reasoning = prediction.reasoning
        except Exception as e:
            predicted_views_str = ""
            reasoning = f"Error: {str(e)}"
        
        # Parse views
        predicted_views = set(parse_view_list(predicted_views_str))
        expected_views = set(parse_view_list(expected_views_str))
        
        # Calculate metrics
        precision = calculate_precision(predicted_views, expected_views)
        recall = calculate_recall(predicted_views, expected_views)
        f1 = calculate_f1(precision, recall)
        accuracy = calculate_accuracy(predicted_views, expected_views)
        
        return {
            'question': question,
            'expected_views': expected_views_str,
            'predicted_views': predicted_views_str,
            'reasoning': reasoning,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'match': accuracy == 1.0
        }
    
    def evaluate_dataset(self, examples: List[Dict[str, Any]], 
                        verbose: bool = True) -> Dict[str, Any]:
        """
        Evaluate the module on a dataset.
        
        Args:
            examples: List of example dictionaries
            verbose: Whether to show progress bar
            
        Returns:
            Dictionary with aggregate metrics and individual results
        """
        results = []
        
        iterator = tqdm(examples, desc="Evaluating") if verbose else examples
        
        for example in iterator:
            result = self.evaluate_example(example)
            results.append(result)
        
        # Calculate aggregate metrics
        avg_precision = sum(r['precision'] for r in results) / len(results)
        avg_recall = sum(r['recall'] for r in results) / len(results)
        avg_f1 = sum(r['f1'] for r in results) / len(results)
        avg_accuracy = sum(r['accuracy'] for r in results) / len(results)
        
        return {
            'results': results,
            'metrics': {
                'precision': avg_precision,
                'recall': avg_recall,
                'f1': avg_f1,
                'accuracy': avg_accuracy,
                'total_examples': len(results),
                'exact_matches': sum(1 for r in results if r['match'])
            }
        }
    