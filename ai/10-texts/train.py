import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer
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
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Construir vocabulario
def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

train_dataset = ReviewDataset(X_train.values, y_train.values, tokenizer)
test_dataset = ReviewDataset(X_test.values, y_test.values, tokenizer)

def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    return input_ids, attention_mask, labels

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Create model
embed_size = 100
hidden_size = 128
num_classes = 2
vocab_size = tokenizer.vocab_size  # Tamaño del vocabulario del tokenizador

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Use device:', device)

model = SentimentModel(vocab_size, embed_size, hidden_size, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        for input_ids, attention_mask, labels in train_loader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

train_model(model, train_loader, criterion, optimizer)

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in test_loader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

evaluate_model(model, test_loader)

torch.save(model.state_dict(), "model.pth")
