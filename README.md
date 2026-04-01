# iSHAP (Informed SHAP) - Research Repo

This is the research repo for the iSHAP paper. The goal is to extend the SHAP framework to be more grounded in domain knowledge thus giving more realistic feature contributions
and then leveraging the SHAP and shapiq findings to generate an explanatory model.
The hypothesis is that LLMs can integrate domain knowledge into post-hoc explanation methods for more realistic and 
grounded explanations which then helps in clinical decision support and scientific discovery.

## Current To Dos

[ ] Validate the evaluation pipeline on other datasets 
[ ] Validate the evaluation pipeline on other models
[ ] Improve synthetic cohort generation
[ ] Integrate causal hypothesis generalization
[ ] Experiment with agentic workflows

## Project Layout

- `scripts/`: Python scripts for preprocessing datasets, ishap functionality, and evaluation pipeline.
- `manuscript/`: LaTeX manuscript for the iSHAP paper.
- `results/`: Evaluation pipeline results.

## Quick Start for running evaluation pipeline

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Make a Mistral account. Put your API key in the `.env` file as MISTRAL_API_KEY.
3. Run scripts/run_mimic_ishap.py for the evaluation pipeline.
