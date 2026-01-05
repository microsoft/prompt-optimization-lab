"""
Main entry point for prompt optimization lab.
"""

import sys
from pathlib import Path

from run_baseline_evaluation import main as run_baseline
from run_pipeline_evaluation import main as run_pipeline

def main():
    print("🔬 Prompt Optimization Lab")
    print("="*60)
    
    #run_baseline()
    run_pipeline()


if __name__ == "__main__":
    main()