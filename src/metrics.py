"""
Evaluation metrics for view selector performance.
"""

from typing import List, Set, Dict, Any
import numpy as np


def calculate_precision(predicted: Set[str], expected: Set[str]) -> float:
    """
    Calculate precision: TP / (TP + FP)
    
    Args:
        predicted: Set of predicted view names
        expected: Set of expected view names
        
    Returns:
        Precision score between 0 and 1
    """
    if not predicted:
        return 0.0
    
    true_positives = len(predicted & expected)
    return true_positives / len(predicted)


def calculate_recall(predicted: Set[str], expected: Set[str]) -> float:
    """
    Calculate recall: TP / (TP + FN)
    
    Args:
        predicted: Set of predicted view names
        expected: Set of expected view names
        
    Returns:
        Recall score between 0 and 1
    """
    if not expected:
        return 0.0
    
    true_positives = len(predicted & expected)
    return true_positives / len(expected)


def calculate_f1(precision: float, recall: float) -> float:
    """
    Calculate F1 score: 2 * (precision * recall) / (precision + recall)
    
    Args:
        precision: Precision score
        recall: Recall score
        
    Returns:
        F1 score between 0 and 1
    """
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)


def calculate_accuracy(predicted: Set[str], expected: Set[str]) -> float:
    """
    Calculate exact match accuracy.
    
    Args:
        predicted: Set of predicted view names
        expected: Set of expected view names
        
    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    return 1.0 if predicted == expected else 0.0


def evaluate_predictions(predictions: List[Dict[str, Any]], 
                        expected: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Evaluate a list of predictions against expected results.
    
    Args:
        predictions: List of prediction dictionaries with 'selected_views'
        expected: List of expected dictionaries with 'expected_views'
        
    Returns:
        Dictionary with average metrics
    """
    precisions = []
    recalls = []
    f1_scores = []
    accuracies = []
    
    for pred, exp in zip(predictions, expected):
        # Parse view lists
        pred_views = set(pred.get('selected_views', '').split(',')) if pred.get('selected_views') else set()
        exp_views = set(exp.get('expected_views', '').split(',')) if exp.get('expected_views') else set()
        
        # Remove empty strings and '<NO_VIEWS>'
        pred_views = {v.strip() for v in pred_views if v.strip() and v.strip() != '<NO_VIEWS>'}
        exp_views = {v.strip() for v in exp_views if v.strip() and v.strip() != '<NO_VIEWS>'}
        
        # Calculate metrics
        precision = calculate_precision(pred_views, exp_views)
        recall = calculate_recall(pred_views, exp_views)
        f1 = calculate_f1(precision, recall)
        accuracy = calculate_accuracy(pred_views, exp_views)
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        accuracies.append(accuracy)
    
    return {
        'precision': np.mean(precisions),
        'recall': np.mean(recalls),
        'f1': np.mean(f1_scores),
        'accuracy': np.mean(accuracies),
        'total_examples': len(predictions)
    }

def strict_view_selector_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """
    Strict evaluation metric (exact match) for DSPy optimization.
    5-argument signature required by GEPA.
    """
    try:
        predicted_views = getattr(pred, 'selected_views', [])
        expected_views = getattr(gold, 'selected_views', [])
        
        if isinstance(predicted_views, str):
            predicted_views = [v.strip() for v in predicted_views.split(',') if v.strip()]
        elif not isinstance(predicted_views, list):
            predicted_views = [str(predicted_views)]
            
        if isinstance(expected_views, str):
            expected_views = [v.strip() for v in expected_views.split(',') if v.strip()]
        elif not isinstance(expected_views, list):
            expected_views = [str(expected_views)]
        
        pred_set = set(predicted_views)
        expected_set = set(expected_views)
        
        return 1.0 if pred_set == expected_set else 0.0
        
    except Exception as e:
        print(f"Error in strict metric: {e}")
        return 0.0


def soft_view_selector_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """
    Soft evaluation metric (Jaccard similarity) for DSPy optimization.
    5-argument signature required by GEPA.
    """
    try:
        predicted_views = getattr(pred, 'selected_views', [])
        expected_views = getattr(gold, 'selected_views', [])
        
        if isinstance(predicted_views, str):
            predicted_views = [v.strip() for v in predicted_views.split(',') if v.strip()]
        elif not isinstance(predicted_views, list):
            predicted_views = [str(predicted_views)]
            
        if isinstance(expected_views, str):
            expected_views = [v.strip() for v in expected_views.split(',') if v.strip()]
        elif not isinstance(expected_views, list):
            expected_views = [str(expected_views)]
        
        pred_set = set(predicted_views)
        expected_set = set(expected_views)

        if not pred_set and not expected_set:
            return 1.0
        if not expected_set:
            return 0.0
        
        intersection = len(pred_set & expected_set)
        union = len(pred_set | expected_set)
        return intersection / union if union > 0 else 0.0
    
    except Exception as e:
        print(f"⚠️ Soft metric error: {e}")
        return 0.0

def calculate_metrics(predicted_views, expected_views) -> Dict[str, float]:
    """Calculate all metrics for a prediction."""
    def normalize_views(views) -> Set[str]:
        if not views:
            return set()
        if isinstance(views, list):
            return set(str(v).strip() for v in views if v and str(v).strip() != '<NO_VIEWS>')
        if isinstance(views, str):
            if views == '<NO_VIEWS>' or not views.strip():
                return set()
            return set(v.strip() for v in views.split(',') if v.strip() and v.strip() != '<NO_VIEWS>')
        return set()
    
    pred_set = normalize_views(predicted_views)
    exp_set = normalize_views(expected_views)
    
    exact_match = 1.0 if pred_set == exp_set else 0.0
    
    if not pred_set and not exp_set:
        return {'exact_match': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'jaccard': 1.0}
    if not exp_set:
        return {'exact_match': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'jaccard': 0.0}
    
    intersection = pred_set & exp_set
    precision = len(intersection) / len(pred_set) if pred_set else 0.0
    recall = len(intersection) / len(exp_set) if exp_set else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    union = pred_set | exp_set
    jaccard = len(intersection) / len(union) if union else 0.0
    
    return {
        'exact_match': exact_match,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'jaccard': jaccard
    }

def print_evaluation_report(metrics: Dict[str, float]):
    """
    Print a formatted evaluation report.
    
    Args:
        metrics: Dictionary of evaluation metrics
    """
    print("\n" + "="*60)
    print("📊 EVALUATION RESULTS")
    print("="*60)
    print(f"Total Examples: {metrics['total_examples']}")
    print(f"Precision:      {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"Recall:         {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"F1 Score:       {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)")
    print(f"Accuracy:       {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print("="*60)
