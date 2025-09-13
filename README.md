NEURAL NETWORK ASSIGNMENT - PROJECT STRUCTURE
=============================================

MAIN EXECUTION FILES:
- main.py              : Run complete pipeline (training + evaluation)
- run_evaluation.py    : Run only evaluation with pre-trained models
- show_results.py      : Display all results summary

SOURCE CODE LOCATION:
- All main code is in the src/ folder:
  - src/model.py       : VAE and AE model definitions
  - src/train.py       : Training procedures
  - src/evaluate.py    : Evaluation metrics (FID, IS)
  - src/utils.py       : Data loading and utilities
  - src/config.py      : All hyperparameters
  - src/visualizations.py : Plot generation
  - src/data_analysis.py  : Dataset analysis

QUICK START:
1. pip install -r requirements.txt
2. python main.py       (complete pipeline)
3. python show_results.py (view results)

