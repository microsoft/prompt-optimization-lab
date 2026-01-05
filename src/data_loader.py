"""
Data loading utilities for golden dataset and evaluation data.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any


def load_golden_dataset(filepath: str = "data/golden_dataset_v1.csv") -> pd.DataFrame:
    """
    Load the golden dataset CSV file.
    
    Args:
        filepath: Path to the golden dataset CSV
        
    Returns:
        DataFrame with golden dataset
    """
    df = pd.read_csv(filepath)
    return df


def load_batch_test_results(filepath: str = "data/batch_test_results_20250924_172439.jsonl") -> List[Dict[str, Any]]:
    """
    Load batch test results from JSONL file.
    
    Args:
        filepath: Path to the batch test results JSONL file
        
    Returns:
        List of test result dictionaries
    """
    results = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def load_snowflake_views(filepath: str = "data/snowflake_view.json") -> List[Dict[str, Any]]:
    """
    Load Snowflake view metadata.
    
    Args:
        filepath: Path to the Snowflake views JSON file
        
    Returns:
        List of view metadata dictionaries
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract the views array from the JSON structure
    if isinstance(data, dict) and 'views' in data:
        views = data['views']
    elif isinstance(data, list):
        views = data
    else:
        views = []
    
    return views


def create_dspy_examples(golden_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert golden dataset to DSPy-compatible examples.
    
    Args:
        golden_df: Golden dataset DataFrame
        
    Returns:
        List of example dictionaries for DSPy
    """
    examples = []
    for _, row in golden_df.iterrows():
        example = {
            'question': row['question'],
            'conversation_history': row.get('conversation_history', ''),
            'expected_views': row['expected_views'],
            'selected_views_actual': row.get('selected_views_actual', '')
        }
        examples.append(example)
    
    return examples


def parse_view_list(view_string: str) -> List[str]:
    """
    Parse a comma-separated view string into a list.
    
    Args:
        view_string: Comma-separated view names or '<NO_VIEWS>'
        
    Returns:
        List of view names
    """
    if not view_string or view_string == '<NO_VIEWS>':
        return []
    
    # Handle both comma and newline separators
    views = view_string.replace('\n', ',').split(',')
    return [v.strip() for v in views if v.strip()]
