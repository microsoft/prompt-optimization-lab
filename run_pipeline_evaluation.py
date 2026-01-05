"""
Run integrated pipeline evaluation (View Selector + SQL Writer).
"""

import sys
import json
from pathlib import Path
import dspy

from src.modules import ViewToSQLPipeline
from src.data_loader import load_golden_dataset, load_snowflake_views, create_dspy_examples
from tqdm import tqdm


def evaluate_pipeline(pipeline, examples, verbose=True):
    """Evaluate the integrated pipeline on examples."""
    results = []
    
    iterator = tqdm(examples, desc="Evaluating Pipeline") if verbose else examples
    
    for example in iterator:
        question = example['question']
        conversation_history = example.get('conversation_history', '')
        expected_views = example['expected_views']
        
        try:
            # Run pipeline
            prediction = pipeline(
                question=question,
                conversation_history=conversation_history
            )
            
            result = {
                'question': question,
                'expected_views': expected_views,
                'selected_views': prediction.selected_views,
                'view_reasoning': prediction.view_reasoning,
                'sql_queries': prediction.sql_queries,
                'sql_reasoning': prediction.reasoning,
                'iterations': prediction.iterations,
                'success': '<NO_QUERIES>' not in prediction.sql_queries
            }
        except Exception as e:
            result = {
                'question': question,
                'expected_views': expected_views,
                'error': str(e),
                'success': False
            }
        
        results.append(result)
    
    return results


def main():
    print("🚀 Starting Integrated Pipeline Evaluation")
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
    
    # Initialize pipeline
    print("\n🔧 Initializing integrated pipeline (ViewSelector + SQLWriter)...")
    pipeline = ViewToSQLPipeline(candidate_views=snowflake_views)
    print("✅ Pipeline initialized")
    
    # Run evaluation on a subset (first 5 examples for testing)
    print("\n🧪 Running pipeline evaluation (first 5 examples)...")
    test_examples = examples
    results = evaluate_pipeline(pipeline, test_examples, verbose=True)
    
    # Calculate success rate
    success_count = sum(1 for r in results if r.get('success', False))
    success_rate = success_count / len(results) * 100
    
    print("\n" + "="*60)
    print("📊 PIPELINE EVALUATION RESULTS")
    print("="*60)
    print(f"Total Examples: {len(results)}")
    print(f"Successful SQL Generation: {success_count} ({success_rate:.2f}%)")
    print("="*60)
    
    # Save results
    output_dir = Path("data/baseline")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "pipeline_evaluation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Print sample results
    print("\n📋 Sample Results:")
    print("="*60)
    for i, result in enumerate(results[:3], 1):
        print(f"\n{'─'*60}")
        print(f"Example {i}:")
        print(f"Question: {result['question'][:80]}...")
        print(f"Selected Views: {result.get('selected_views', 'N/A')[:80]}...")
        if result.get('success'):
            print(f"SQL Queries: {result.get('sql_queries', 'N/A')[:150]}...")
            print(f"✅ Success")
        else:
            print(f"❌ Failed: {result.get('error', 'No SQL generated')}")
        print(f"{'─'*60}")


if __name__ == "__main__":
    main()