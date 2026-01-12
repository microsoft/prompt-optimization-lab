"""
Optimization script for IntegratedText2SQLModule using DSPy optimizers.

Usage:
    python scripts/optimize_integrated_text2sql.py --optimizer labeledfewshot
    python scripts/optimize_integrated_text2sql.py --optimizer gepa --train-size 15
    python scripts/optimize_integrated_text2sql.py --optimizer bootstrap --max-demos 8
    python scripts/optimize_integrated_text2sql.py --optimizer all
"""

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Dict, Any, List

import dspy
from dspy.teleprompt import LabeledFewShot, GEPA, BootstrapFewShot
import pandas as pd

from src.modules_v2 import IntegratedText2SQLModule
from src.data_loader import load_golden_dataset, load_snowflake_views, create_dspy_examples
from src.metrics import (
    strict_view_selector_metric, 
    soft_view_selector_metric,
    calculate_metrics,
    print_evaluation_report
)
from src.optimizer.base import OptimizerFactory

class IntegratedText2SQLEvaluator:
    """Evaluator for IntegratedText2SQLModule."""
    
    def __init__(self, module: IntegratedText2SQLModule):
        self.module = module
    
    def evaluate_example(self, example: dict) -> dict:
        """Evaluate a single example through the two-stage pipeline."""
        question = example['question']
        conversation_history = example.get('conversation_history', '')
        expected_views_str = example['expected_views']
        
        # Run the integrated pipeline
        try:
            prediction = self.module(
                question=question,
                conversation_history=conversation_history
            )
            
            predicted_views_str = getattr(prediction, 'selected_views', '<NO_VIEWS>')
            resolved_entities_str = getattr(prediction, 'resolved_entities', '{}')
            
        except Exception as e:
            predicted_views_str = '<NO_VIEWS>'
            resolved_entities_str = '{}'
        
        # Parse views
        def parse_views(views_str):
            if not views_str or views_str == '<NO_VIEWS>':
                return set()
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
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'accuracy': metrics['exact_match'],
            'jaccard': metrics['jaccard'],
            'match': metrics['exact_match'] == 1.0
        }
    
    def evaluate_dataset(self, examples: list, verbose: bool = True) -> dict:
        """Evaluate the module on a dataset."""
        from tqdm import tqdm
        
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


class IntegratedText2SQLOptimizer:
    """Main optimizer for IntegratedText2SQLModule."""
    
    def __init__(self, output_dir: Path = Path("data/optimization_results_v2")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.modules_dir = Path("data/optimized_modules_v2")
        self.modules_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure DSPy
        self.main_lm = dspy.LM(model="azure/gpt-4o", temperature=0.0, max_tokens=2000)
        dspy.configure(lm=self.main_lm)
        
        self.reflection_lm = dspy.LM(model="azure/gpt-4.1", temperature=1.0, max_tokens=10000)
        self.optimizer_factory = OptimizerFactory()
        
        # Load data
        self.snowflake_views = load_snowflake_views()
        self.golden_df = load_golden_dataset()
        self.examples = create_dspy_examples(self.golden_df)
        
        # Convert to DSPy format
        self.dspy_examples = self._prepare_dspy_examples()
        
        print(f"✅ Initialized IntegratedText2SQLOptimizer")
        print(f"   • Loaded {len(self.snowflake_views)} views")
        print(f"   • Loaded {len(self.examples)} examples")
    
    def _prepare_dspy_examples(self):
        """Convert examples to DSPy format."""
        dspy_examples = []
        
        for ex in self.examples:
            expected_views = ex.get('expected_views', '')
            if isinstance(expected_views, str):
                if expected_views.strip() == '<NO_VIEWS>' or not expected_views.strip():
                    views_list = ['<NO_VIEWS>']
                else:
                    views_list = [v.strip() for v in expected_views.split(',') if v.strip()]
            else:
                views_list = expected_views if isinstance(expected_views, list) else []
            
            dspy_example = dspy.Example(
                question=ex['question'],
                conversation_history=ex.get('conversation_history', ''),
                selected_views=views_list
            ).with_inputs('question', 'conversation_history')
            
            dspy_examples.append(dspy_example)
        
        return dspy_examples
    
    def split_data(self, train_ratio: float = 0.7):
        """Split data into train and validation sets."""
        import random
        shuffled = self.dspy_examples.copy()
        random.shuffle(shuffled)
        
        split_idx = int(len(shuffled) * train_ratio)
        return shuffled[:split_idx], shuffled[split_idx:]
    
    def optimize_labeledfewshot(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run LabeledFewShot optimization."""
        print("\n🚀 LABELEDFEWSHOT OPTIMIZATION (IntegratedText2SQL)")
        print("=" * 60)
        
        train_set, val_set = self.split_data()
        
        # Create fresh module
        student = IntegratedText2SQLModule(config={}, candidate_views=self.snowflake_views)
        # Create optimizer
        optimizer = self.optimizer_factory.create_labeledfewshot(config)
        
        # Compile
        print("🔧 Compiling...")
        start_time = time.time()
        optimized_module = optimizer.compile(student=student, trainset=train_set)
        compilation_time = time.time() - start_time
        
        print(f"✅ Completed in {compilation_time:.2f}s")
        
        # Evaluate
        results = self._evaluate_module(optimized_module, "labeledfewshot")
        results['compilation_time'] = compilation_time
        
        # Save
        self._save_results(optimized_module, results, "labeledfewshot")
        
        return results
    
    def optimize_gepa(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run GEPA optimization."""
        print("\n🧠 GEPA OPTIMIZATION (IntegratedText2SQL)")
        print("=" * 60)
        
        train_set, val_set = self.split_data()
        
        # Limit train/val size for GEPA
        train_size = config.get('train_size', 15)
        val_size = config.get('val_size', 10)
        train_set = train_set[:train_size]
        val_set = val_set[:val_size]
        
        # Create fresh module
        student = IntegratedText2SQLModule(config={}, candidate_views=self.snowflake_views)
        
        # Create optimizer
        optimizer = self.optimizer_factory.create_gepa(config, self.reflection_lm)
        
        # Compile
        print("🔧 Compiling (this may take several minutes)...")
        start_time = time.time()
        
        try:
            optimized_module = optimizer.compile(
                student=student,
                trainset=train_set,
                valset=val_set
            )
            compilation_time = time.time() - start_time
            print(f"✅ Completed in {compilation_time:.2f}s ({compilation_time/60:.1f} minutes)")
            
            # Evaluate
            results = self._evaluate_module(optimized_module, "gepa")
            results['compilation_time'] = compilation_time
            
            # Save
            self._save_results(optimized_module, results, "gepa")
            
            return results
            
        except Exception as e:
            print(f"❌ GEPA optimization failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def optimize_bootstrap(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run BootstrapFewShot optimization."""
        print("\n🔄 BOOTSTRAPFEWSHOT OPTIMIZATION (IntegratedText2SQL)")
        print("=" * 60)
        
        train_set, _ = self.split_data()
        
        # Create fresh module
        student = IntegratedText2SQLModule(config={}, candidate_views=self.snowflake_views)
        
        # Create optimizer
        optimizer = self.optimizer_factory.create_bootstrap(config)
        
        # Compile
        print("🔧 Compiling...")
        start_time = time.time()
        
        try:
            optimized_module = optimizer.compile(student=student, trainset=train_set)
            compilation_time = time.time() - start_time
            print(f"✅ Completed in {compilation_time:.2f}s")
            
            # Evaluate
            results = self._evaluate_module(optimized_module, "bootstrap")
            results['compilation_time'] = compilation_time
            
            # Save
            self._save_results(optimized_module, results, "bootstrap")
            
            return results
            
        except Exception as e:
            print(f"❌ Bootstrap optimization failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _evaluate_module(self, module, optimizer_name: str) -> Dict[str, Any]:
        """Evaluate optimized module."""
        print(f"\n📊 Evaluating {optimizer_name} module...")
        
        evaluator = IntegratedText2SQLEvaluator(module)
        evaluation_results = evaluator.evaluate_dataset(self.examples, verbose=True)
        
        results = {
            'optimizer': optimizer_name,
            'metrics': evaluation_results['metrics'],
            'predictions': evaluation_results['results']
        }
        
        print_evaluation_report(evaluation_results['metrics'])
        
        return results
    
    def _save_results(self, module, results: Dict[str, Any], optimizer_name: str):
        """Save module and results."""
        # Save module
        module_path = self.modules_dir / f"{optimizer_name}_integrated_module.pkl"
        try:
            with open(module_path, 'wb') as f:
                pickle.dump(module, f)
            print(f"✅ Saved module to: {module_path}")
        except Exception as e:
            print(f"⚠️ Could not save module: {e}")
        
        # Save results
        results_path = self.output_dir / f"{optimizer_name}_integrated_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"✅ Saved results to: {results_path}")
    
    def run_all_optimizers(self, configs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Run all optimizers and compare results."""
        print("\n🏁 RUNNING ALL OPTIMIZERS (IntegratedText2SQL)")
        print("=" * 60)
        
        all_results = {}
        
        # LabeledFewShot
        lfs_config = configs.get('labeledfewshot', {'k': 10})
        lfs_results = self.optimize_labeledfewshot(lfs_config)
        if lfs_results:
            all_results['labeledfewshot'] = lfs_results
        
        # GEPA
        gepa_config = configs.get('gepa', {'train_size': 15, 'val_size': 10})
        gepa_results = self.optimize_gepa(gepa_config)
        if gepa_results:
            all_results['gepa'] = gepa_results
        
        # Bootstrap
        bootstrap_config = configs.get('bootstrap', {
            'max_bootstrapped_demos': 8,
            'max_labeled_demos': 8,
            'max_rounds': 5
        })
        bootstrap_results = self.optimize_bootstrap(bootstrap_config)
        if bootstrap_results:
            all_results['bootstrap'] = bootstrap_results
        
        # Generate comparison
        self._generate_comparison(all_results)
        
        return all_results
    
    def _generate_comparison(self, all_results: Dict[str, Any]):
        """Generate comparison report."""
        print("\n📊 COMPREHENSIVE COMPARISON (IntegratedText2SQL)")
        print("=" * 60)
        
        comparison_data = []
        for optimizer, results in all_results.items():
            metrics = results.get('metrics', {})
            comparison_data.append({
                'optimizer': optimizer,
                'accuracy': metrics.get('accuracy', 0.0),
                'precision': metrics.get('precision', 0.0),
                'recall': metrics.get('recall', 0.0),
                'f1': metrics.get('f1', 0.0),
                'jaccard': metrics.get('jaccard', 0.0),
                'compilation_time': results.get('compilation_time', 0.0)
            })
        
        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))
        
        # Save comparison
        comparison_path = self.output_dir / "optimization_comparison_integrated.csv"
        df.to_csv(comparison_path, index=False)
        print(f"\n✅ Saved comparison to: {comparison_path}")
        
        # Find best
        if len(df) > 0:
            best_idx = df['accuracy'].idxmax()
            best_optimizer = df.loc[best_idx, 'optimizer']
            best_accuracy = df.loc[best_idx, 'accuracy']
            
            print(f"\n🏆 Best Optimizer: {best_optimizer}")
            print(f"   Accuracy: {best_accuracy:.3f} ({best_accuracy*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Optimize IntegratedText2SQLModule")
    parser.add_argument(
        '--optimizer',
        choices=['labeledfewshot', 'gepa', 'bootstrap', 'all'],
        required=True,
        help='Optimizer to use'
    )
    parser.add_argument('--k', type=int, default=10, help='k for LabeledFewShot')
    parser.add_argument('--train-size', type=int, default=15, help='Train size for GEPA')
    parser.add_argument('--val-size', type=int, default=10, help='Val size for GEPA')
    parser.add_argument('--max-demos', type=int, default=8, help='Max demos for Bootstrap')
    parser.add_argument('--output-dir', type=str, default='data/optimization_results_v2',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer_runner = IntegratedText2SQLOptimizer(output_dir=Path(args.output_dir))
    
    # Run optimization
    if args.optimizer == 'labeledfewshot':
        config = {'k': args.k}
        optimizer_runner.optimize_labeledfewshot(config)
    
    elif args.optimizer == 'gepa':
        config = {'train_size': args.train_size, 'val_size': args.val_size}
        optimizer_runner.optimize_gepa(config)
    
    elif args.optimizer == 'bootstrap':
        config = {
            'max_bootstrapped_demos': args.max_demos,
            'max_labeled_demos': args.max_demos,
            'max_rounds': 5
        }
        optimizer_runner.optimize_bootstrap(config)
    
    elif args.optimizer == 'all':
        configs = {
            'labeledfewshot': {'k': args.k},
            'gepa': {'train_size': args.train_size, 'val_size': args.val_size},
            'bootstrap': {
                'max_bootstrapped_demos': args.max_demos,
                'max_labeled_demos': args.max_demos,
                'max_rounds': 5
            }
        }
        optimizer_runner.run_all_optimizers(configs)
    
    print("\n✅ Optimization complete!")


if __name__ == "__main__":
    main()
    