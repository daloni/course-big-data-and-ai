import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import re

from ReviewDataset import ReviewDataset
from SentimentModel import SentimentModel

df = pd.read_csv('./data/IMDB_Dataset.csv')

def preprocess_text(text):
    text = re.sub(r'\W', ' ', str(text))
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text

df['review'] = df['review'].apply(preprocess_text)

label_encoder = LabelEncoder()
df['sentiment'] = label_encoder.fit_transform(df['sentiment'])

# Tokenizador y vocabulario
tokenizer = get_tokenizer("basic_english")

# Construir vocabulario
def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(df['review']), specials=["<unk>", "<pad>"])
vocab.set_default_index(vocab["<unk>"])

X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

train_dataset = ReviewDataset(X_train.values, y_train.values, vocab)
test_dataset = ReviewDataset(X_test.values, y_test.values, vocab)

def collate_fn(batch):
    reviews, labels = zip(*batch)
    reviews_padded = pad_sequence(reviews, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return reviews_padded, labels

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Create model
embed_size = 100
hidden_size = 128
num_classes = 2
max_len = 100
vocab_size = len(vocab)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Use device:', device)

model = SentimentModel(vocab_size, embed_size, hidden_size, num_classes, max_len).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0
        correct = 0
        total = 0
        for reviews, labels in train_loader:
            reviews, labels = reviews.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(reviews)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, Accuracy: {accuracy}%')

train_model(model, train_loader, criterion, optimizer)

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for reviews, labels in test_loader:
            reviews, labels = reviews.to(device), labels.to(device)

            outputs = model(reviews)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy}%')

evaluate_model(model, test_loader)

torch.save(model.state_dict(), "model.pth")
