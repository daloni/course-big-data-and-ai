import torch.nn as nn

class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes, max_len):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.max_len = max_len
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        last_hidden = h_n[-1, :, :]
        out = self.fc(last_hidden)
        return out