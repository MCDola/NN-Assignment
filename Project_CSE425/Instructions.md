# Open new command prompt/terminal
conda create -n test_env python=3.9 -y
conda activate test_env
pip install -r requirements.txt
python main.py

#Option - A
# 1. Create environment
conda create -n nn_project python=3.9 -y
conda activate nn_project

# 2. Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torchmetrics torch-fidelity scikit-learn matplotlib numpy

# 3. Run complete pipeline
python main.py

# 4. View results
python show_results.py

#Option - B
# 1. Environment setup
conda create -n nn_project python=3.9 -y
conda activate nn_project

# 2. Install packages one by one
pip install torch==1.11.0
pip install torchvision==0.12.0
pip install torchmetrics==0.9.0
pip install torch-fidelity==0.3.0
pip install scikit-learn==1.0.2
pip install matplotlib==3.5.3
pip install numpy==1.22.4

# 3. Run data analysis
python -c "import sys; sys.path.append('src'); from data_analysis import analyze_dataset; analyze_dataset()"

# 4. Train models
python -c "import sys; sys.path.append('src'); from train import train_vae, train_ae; train_vae(); train_ae()"

# 5. Evaluate models
python run_evaluation.py

# 6. Generate visualizations
python -c "import sys; sys.path.append('src'); from visualizations import *; create_training_plots(); create_metrics_comparison(); create_real_latent_space_visualization()"

# 7. Show results
python show_results.py

=====Comprehensive Error Handling Guide=====
# Check if environment is activated
conda activate nn_project

# Check if package is installed
pip list | grep torch

# Install missing package
pip install missing-package-name
