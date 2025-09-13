import torch

batch_size = 128
epochs = 20
learning_rate = 1e-3
latent_dim = 20
beta = 1.0

# Data settings
data_path = './data'
image_size = 28
num_channels = 1

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
log_interval = 100

# Output paths
model_save_path = './outputs/models'
figure_save_path = './outputs/figures'
metrics_save_path = './outputs/metrics'