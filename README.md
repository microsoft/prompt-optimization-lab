# View Selector Optimization with DSPy

## 📚 Project Overview

This project demonstrates **automated prompt optimization** for a database view selector agent using DSPy (Declarative Self-improving Language Programs). The goal is to transform a handwritten prompt-based system into an optimized, data-driven solution that automatically learns to select the most relevant Snowflake database views based on user queries.

### Business Context

**Domain**: Financial data analysis for Private Equity, Real Estate, Infrastructure, Credit, and other investment platforms.

**Problem**: Given a natural language question about financial data (e.g., "What is MIC's current exposure in the USA in PE?"), the system must:
1. Understand the question context and domain terminology
2. Select the most relevant Snowflake database views from 20+ available views
3. Handle financial classification rules (Asset Classes, Investment Classes, Platforms)
4. Maintain conversation context for follow-up questions

**Challenge**: Handwritten prompts are difficult to maintain, don't learn from mistakes, and struggle with edge cases.

---

## 🎯 Project Objectives

### Primary Goal
**Optimize prompt engineering through automated learning** instead of manual prompt crafting.

### Specific Objectives

1. **Baseline Performance**: Establish handwritten prompt performance metrics
2. **DSPy Implementation**: Convert prompt logic into DSPy modules with Chain-of-Thought reasoning
3. **Prompt Optimization**: Apply multiple DSPy optimizers to improve accuracy
4. **Performance Comparison**: Quantify improvements using precision, recall, F1, and accuracy metrics
5. **Production Readiness**: Create deployable optimized modules

## 🚀 Installation & Setup

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer and environment manager

### Environment Initialization

1. **Initialize the project** (if starting fresh):
   ```powershell
   uv init .

2. Add development dependencies for Jupyter notebooks:
    ```powershell
    uv add --dev jupyter ipykernel

## Project Structure
    prompt-optimization-lab/
    ├── README.md
    ├── data/
    │   ├── snowflake_metadata.yaml    # View metadata and descriptions
    │   └── snowflake_view.json        # Structured view definitions
    ├── notebooks/
    │   └── 01_data_preparation.ipynb  # Data loading and exploration
    └── .venv/ 

## Usage
# Run single optimizer
uv run python scripts/optimize_view_selector.py --optimizer labeledfewshot --k 10

# Run GEPA with custom sizes
uv run python src/optimize_view_selector.py --optimizer gepa --train-size 20 --val-size 10

# Run Bootstrap
uv run python scripts/optimize_view_selector.py --optimizer bootstrap --max-demos 10

# Run all optimizers
uv run python scripts/optimize_view_selector.py --optimizer all

# Custom output directory
uv run python scripts/optimize_view_selector.py --optimizer all --output-dir results/exp1