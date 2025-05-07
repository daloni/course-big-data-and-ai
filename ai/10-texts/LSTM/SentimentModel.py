import torch.nn as nn
import torch

class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes, num_layers):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.5, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        masked_embedded = embedded * attention_mask.unsqueeze(-1)
        _, (hidden, _) = self.lstm(masked_embedded)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(hidden)