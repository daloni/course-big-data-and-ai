import torch

CONFIG = {
    'batch_size': 32,
    'embed_size': 128,
    'hidden_size': 256,
    'num_classes': 2,
    'max_len': 200,
    'num_layers': 10,
    'learning_rate': 0.0005,
    'num_epochs': 10,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'model_path': './model.pth',
    'dataset_path': './data/IMDB_Dataset.csv'
}
