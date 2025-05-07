import torch

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'

print(f"Using device: {device}")

CONFIG = {
    'batch_size': 20,
    'embed_size': 100,
    'hidden_size': 200,
    'num_classes': 2,
    'max_len': 500,
    'num_layers': 100,
    'learning_rate': 0.0005,
    'num_epochs': 10,
    'device': device,
    'model_path': './model.pth',
    'dataset_path': './data/IMDB_Dataset.csv'
}