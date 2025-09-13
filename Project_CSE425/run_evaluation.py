import sys
import os

# Add the current directory to the path
sys.path.append(os.getcwd())

# Now import and run the evaluation
from src.full_evaluation import run_complete_evaluation

if __name__ == "__main__":
    run_complete_evaluation()