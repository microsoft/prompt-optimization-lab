"""
Run baseline evaluation on the golden dataset.
"""

import sys
import json
from pathlib import Path
import dspy

from src.modules import ViewSelectorModule
from src.data_loader import load_golden_dataset, load_snowflake_views, create_dspy_examples
from src.evaluator import ViewSelectorEvaluator
from src.metrics import print_evaluation_report


def main():
    print("🚀 Starting Baseline Evaluation")
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
    
    # Initialize baseline module
    print("\n🔧 Initializing baseline ViewSelector module...")
    baseline_module = ViewSelectorModule(candidate_views=snowflake_views)
    print("✅ Module initialized")
    
    # Create evaluator
    evaluator = ViewSelectorEvaluator(baseline_module)
    
    # Run evaluation
    print("\n🧪 Running evaluation...")
    evaluation_results = evaluator.evaluate_dataset(examples, verbose=True)
    
    # Print results
    print_evaluation_report(evaluation_results['metrics'])
    
    # Save results
    output_dir = Path("data/baseline")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "baseline_evaluation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Print sample results
    print("\n📋 Sample Results (first 3):")
    print("="*60)
    for i, result in enumerate(evaluation_results['results'][:3], 1):
        print(f"\nExample {i}:")
        print(f"Question: {result['question'][:100]}...")
        print(f"Expected: {result['expected_views']}")
        print(f"Predicted: {result['predicted_views']}")
        print(f"Match: {'✅' if result['match'] else '❌'}")
        print(f"F1: {result['f1']:.4f}")


if __name__ == "__main__":
    main()