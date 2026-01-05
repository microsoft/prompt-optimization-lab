"""
Run baseline evaluation for IntegratedText2SQLModule (Entity Resolution + View Selection).
This evaluates the two-stage pipeline on the golden dataset.
"""

import sys
import json
from pathlib import Path
import dspy

from src.modules_v2 import IntegratedText2SQLModule
from src.data_loader import load_golden_dataset, load_snowflake_views, create_dspy_examples
from src.metrics import calculate_metrics, print_evaluation_report
from tqdm import tqdm


class IntegratedText2SQLEvaluator:
    """Evaluator for IntegratedText2SQLModule (two-stage pipeline)."""
    
    def __init__(self, module: IntegratedText2SQLModule):
        self.module = module
    
    def evaluate_example(self, example: dict) -> dict:
        """
        Evaluate a single example through the two-stage pipeline.
        
        Args:
            example: Dictionary with 'question', 'expected_views', etc.
            
        Returns:
            Dictionary with prediction results and metrics
        """
        question = example['question']
        conversation_history = example.get('conversation_history', '')
        expected_views_str = example['expected_views']
        
        # Run the integrated pipeline (Entity Resolution → View Selection)
        try:
            prediction = self.module(
                question=question,
                conversation_history=conversation_history
            )
            
            # Extract outputs
            predicted_views_str = getattr(prediction, 'selected_views', '<NO_VIEWS>')
            resolved_entities_str = getattr(prediction, 'resolved_entities', '{}')
            entity_reasoning = getattr(prediction, 'entity_reasoning', '')
            view_reasoning = getattr(prediction, 'view_reasoning', '')
            combined_reasoning = getattr(prediction, 'combined_reasoning', '')
            
        except Exception as e:
            predicted_views_str = '<NO_VIEWS>'
            resolved_entities_str = '{}'
            entity_reasoning = f"Error: {str(e)}"
            view_reasoning = ''
            combined_reasoning = f"Error during prediction: {str(e)}"
        
        # Parse views for comparison
        def parse_views(views_str):
            if not views_str or views_str == '<NO_VIEWS>':
                return set()
            # Handle comma-separated or comma-space separated
            views = views_str.replace('\n', ',').split(',')
            return {v.strip() for v in views if v.strip() and v.strip() != '<NO_VIEWS>'}
        
        predicted_views = parse_views(predicted_views_str)
        expected_views = parse_views(expected_views_str)
        
        # Calculate metrics
        metrics = calculate_metrics(list(predicted_views), list(expected_views))
        
        return {
            'question': question,
            'expected_views': expected_views_str,
            'predicted_views': predicted_views_str,
            'resolved_entities': resolved_entities_str,
            'entity_reasoning': entity_reasoning,
            'view_reasoning': view_reasoning,
            'combined_reasoning': combined_reasoning,
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'accuracy': metrics['exact_match'],
            'jaccard': metrics['jaccard'],
            'match': metrics['exact_match'] == 1.0
        }
    
    def evaluate_dataset(self, examples: list, verbose: bool = True) -> dict:
        """
        Evaluate the module on a dataset.
        
        Args:
            examples: List of example dictionaries
            verbose: Whether to show progress bar
            
        Returns:
            Dictionary with aggregate metrics and individual results
        """
        results = []
        
        iterator = tqdm(examples, desc="Evaluating IntegratedText2SQL") if verbose else examples
        
        for example in iterator:
            result = self.evaluate_example(example)
            results.append(result)
        
        # Calculate aggregate metrics
        avg_precision = sum(r['precision'] for r in results) / len(results) if results else 0.0
        avg_recall = sum(r['recall'] for r in results) / len(results) if results else 0.0
        avg_f1 = sum(r['f1'] for r in results) / len(results) if results else 0.0
        avg_accuracy = sum(r['accuracy'] for r in results) / len(results) if results else 0.0
        avg_jaccard = sum(r['jaccard'] for r in results) / len(results) if results else 0.0
        
        return {
            'results': results,
            'metrics': {
                'precision': avg_precision,
                'recall': avg_recall,
                'f1': avg_f1,
                'accuracy': avg_accuracy,
                'jaccard': avg_jaccard,
                'total_examples': len(results),
                'exact_matches': sum(1 for r in results if r['match'])
            }
        }


def main():
    print("🚀 Starting IntegratedText2SQL Baseline Evaluation (v2)")
    print("="*60)
    
    # Configure DSPy
    print("\n📡 Configuring DSPy with Azure OpenAI...")
    lm = dspy.LM("azure/gpt-4o")
    dspy.configure(lm=lm)
    print("✅ DSPy configured")
    
    # Load data
    print("\n📂 Loading data...")
    golden_df = load_golden_dataset()
    snowflake_views = load_snowflake_views()
    examples = create_dspy_examples(golden_df)
    print(f"✅ Loaded {len(examples)} examples and {len(snowflake_views)} views")
    
    # Initialize IntegratedText2SQL module
    print("\n🔧 Initializing IntegratedText2SQLModule (Entity Resolver + View Selector)...")
    config = {}  # Add any config needed
    baseline_module_v2 = IntegratedText2SQLModule(
        config=config,
        candidate_views=snowflake_views
    )
    print("✅ Module initialized")
    print("   • Stage 1: Entity Resolution")
    print("   • Stage 2: View Selection")
    
    # Create evaluator
    evaluator = IntegratedText2SQLEvaluator(baseline_module_v2)
    
    # Run evaluation
    print("\n🧪 Running evaluation...")
    evaluation_results = evaluator.evaluate_dataset(examples, verbose=True)
    
    # Print results
    print_evaluation_report(evaluation_results['metrics'])
    
    # Save results
    output_dir = Path("data/baseline_v2")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "integrated_baseline_evaluation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Print sample results with entity resolution
    print("\n📋 Sample Results (first 3):")
    print("="*60)
    for i, result in enumerate(evaluation_results['results'][:3], 1):
        print(f"\n{'─'*60}")
        print(f"Example {i}:")
        print(f"Question: {result['question'][:100]}...")
        print(f"\n🔍 Stage 1 - Entity Resolution:")
        print(f"   Resolved Entities: {result['resolved_entities']}")
        print(f"\n🎯 Stage 2 - View Selection:")
        print(f"   Expected Views: {result['expected_views']}")
        print(f"   Predicted Views: {result['predicted_views']}")
        print(f"   Match: {'✅' if result['match'] else '❌'}")
        print(f"   F1: {result['f1']:.4f}")
        print(f"   Precision: {result['precision']:.4f}")
        print(f"   Recall: {result['recall']:.4f}")
    
    # Additional analysis: Entity resolution impact
    print("\n📊 Entity Resolution Analysis:")
    print("="*60)
    
    entity_resolution_stats = {
        'total_examples': len(evaluation_results['results']),
        'examples_with_entities': 0,
        'examples_without_entities': 0,
        'avg_entities_per_query': 0
    }
    
    total_entities = 0
    for result in evaluation_results['results']:
        try:
            entities = json.loads(result['resolved_entities'])
            num_entities = len(entities)
            total_entities += num_entities
            
            if num_entities > 0:
                entity_resolution_stats['examples_with_entities'] += 1
            else:
                entity_resolution_stats['examples_without_entities'] += 1
        except:
            entity_resolution_stats['examples_without_entities'] += 1
    
    entity_resolution_stats['avg_entities_per_query'] = (
        total_entities / entity_resolution_stats['total_examples']
    )
    
    print(f"Total Examples: {entity_resolution_stats['total_examples']}")
    print(f"Examples with Entities: {entity_resolution_stats['examples_with_entities']}")
    print(f"Examples without Entities: {entity_resolution_stats['examples_without_entities']}")
    print(f"Average Entities per Query: {entity_resolution_stats['avg_entities_per_query']:.2f}")
    
    # Save entity resolution stats
    stats_file = output_dir / "entity_resolution_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(entity_resolution_stats, f, indent=2)
    
    print(f"\n💾 Entity stats saved to: {stats_file}")
    
    print("\n" + "="*60)
    print("✅ Evaluation complete!")
    print("="*60)


if __name__ == "__main__":
    main()