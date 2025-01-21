import torch.nn as nn

class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        _, (hidden, _) = self.lstm(embedded)
        out = self.fc(hidden[-1])
        return out