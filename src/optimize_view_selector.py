"""
Main script for optimizing ViewSelectorModule using DSPy optimizers.

Usage:
    python scripts/optimize_view_selector.py --optimizer labeledfewshot
    python scripts/optimize_view_selector.py --optimizer gepa --train-size 15
    python scripts/optimize_view_selector.py --optimizer bootstrap --max-demos 8
    python scripts/optimize_view_selector.py --optimizer all
"""

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Dict, Any, Optional

import dspy
from dspy.teleprompt import LabeledFewShot, GEPA, BootstrapFewShot

from src.modules import ViewSelectorModule
from src.data_loader import load_golden_dataset, load_snowflake_views, create_dspy_examples
from src.metrics import (
    strict_view_selector_metric, 
    soft_view_selector_metric,
    calculate_metrics,
    print_evaluation_report
)
from src.evaluator import ViewSelectorEvaluator


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


class ViewSelectorOptimizer:
    """Main optimizer for ViewSelectorModule."""
    
    def __init__(self, output_dir: Path = Path("data/optimization_results")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.modules_dir = Path("data/optimized_modules")
        self.modules_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure DSPy
        self.main_lm = dspy.LM(model="azure/gpt-4o", temperature=0.0, max_tokens=2000)
        dspy.configure(lm=self.main_lm)
        
        self.reflection_lm = dspy.LM(model="azure/gpt-4.1", temperature=1.0, max_tokens=10000)
        
        # Load data
        self.snowflake_views = load_snowflake_views()
        self.golden_df = load_golden_dataset()
        self.examples = create_dspy_examples(self.golden_df)
        
        # Convert to DSPy format
        self.dspy_examples = self._prepare_dspy_examples()
        
        print(f"✅ Initialized ViewSelectorOptimizer")
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
        print("\n🚀 LABELEDFEWSHOT OPTIMIZATION")
        print("=" * 60)
        
        train_set, val_set = self.split_data()
        
        # Create fresh module
        student = ViewSelectorModule(candidate_views=self.snowflake_views)
        
        # Create optimizer
        optimizer = OptimizerFactory.create_labeledfewshot(config)
        
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
        print("\n🧠 GEPA OPTIMIZATION")
        print("=" * 60)
        
        train_set, val_set = self.split_data()
        
        # Limit train/val size for GEPA
        train_size = config.get('train_size', 15)
        val_size = config.get('val_size', 10)
        train_set = train_set[:train_size]
        val_set = val_set[:val_size]
        
        # Create fresh module
        student = ViewSelectorModule(candidate_views=self.snowflake_views)
        
        # Create optimizer
        optimizer = OptimizerFactory.create_gepa(config, self.reflection_lm)
        
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
        print("\n🔄 BOOTSTRAPFEWSHOT OPTIMIZATION")
        print("=" * 60)
        
        train_set, _ = self.split_data()
        
        # Create fresh module
        student = ViewSelectorModule(candidate_views=self.snowflake_views)
        
        # Create optimizer
        optimizer = OptimizerFactory.create_bootstrap(config)
        
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
        
        evaluator = ViewSelectorEvaluator(module)
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
        module_path = self.modules_dir / f"{optimizer_name}_module.pkl"
        try:
            with open(module_path, 'wb') as f:
                pickle.dump(module, f)
            print(f"✅ Saved module to: {module_path}")
        except Exception as e:
            print(f"⚠️ Could not save module: {e}")
        
        # Save results
        results_path = self.output_dir / f"{optimizer_name}_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"✅ Saved results to: {results_path}")
    
    def run_all_optimizers(self, configs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Run all optimizers and compare results."""
        print("\n🏁 RUNNING ALL OPTIMIZERS")
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
        import pandas as pd
        
        print("\n📊 COMPREHENSIVE COMPARISON")
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
                'compilation_time': results.get('compilation_time', 0.0)
            })
        
        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))
        
        # Save comparison
        comparison_path = self.output_dir / "optimization_comparison.csv"
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
    parser = argparse.ArgumentParser(description="Optimize ViewSelectorModule")
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
    parser.add_argument('--output-dir', type=str, default='data/optimization_results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer_runner = ViewSelectorOptimizer(output_dir=Path(args.output_dir))
    
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
