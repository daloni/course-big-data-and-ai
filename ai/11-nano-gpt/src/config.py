# GPT_CONFIG_124M = {
#     "vocab_size": 50257,     # Vocabulary size
#     "context_length": 256,   # Shortened context length (orig: 1024)
#     "emb_dim": 768,          # Embedding dimension
#     "n_heads": 12,           # Number of attention heads
#     "n_layers": 12,          # Number of layers
#     "drop_rate": 0.1,        # Dropout rate
#     "qkv_bias": False        # Query-Key-Value bias
# }

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1024,
    "n_heads": 16,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

OTHER_SETTINGS = {
    "learning_rate": 5e-4,
    "num_epochs": 5,
    "batch_size": 1,
    "weight_decay": 0.1
}
