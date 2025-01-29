import torch

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'

print(f"Using device: {device}")

CONFIG = {
    'batch_size': 32,
    'max_len': 256,
    'learning_rate': 2e-5,
    'num_epochs': 3,
    'device': device,
    'model_path': './model.pth',
    'dataset_path': './data/IMDB_Dataset.csv'
}
