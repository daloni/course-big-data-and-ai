import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import re
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from tqdm import tqdm

from ReviewDataset import ReviewDataset
from SentimentModel import SentimentModel
from config import CONFIG

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"(\W)(?=\S)", r" \1 ", text)
    text = re.sub(r"'\s", " ", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    return input_ids, attention_mask, labels

df = pd.read_csv(CONFIG['dataset_path'])
df['review'] = df['review'].apply(preprocess_text)
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=20)

train_dataset = ReviewDataset(X_train.values, y_train.values, tokenizer, CONFIG['max_len'])
test_dataset = ReviewDataset(X_test.values, y_test.values, tokenizer, CONFIG['max_len'])

train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=collate_fn)

def train_model(model, train_loader, criterion, optimizer, scheduler, device, num_epochs=5, clip=1.0, accumulation_steps=4):
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        optimizer.zero_grad()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        # for input_ids, attention_mask, labels in progress_bar:
        for i, (input_ids, attention_mask, labels) in enumerate(progress_bar):
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            progress_bar.set_postfix(loss=total_loss/len(train_loader), accuracy=100*correct/total)

        scheduler.step()

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for input_ids, attention_mask, labels in test_loader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    print(classification_report(all_labels, all_preds, target_names=['Negative', 'Positive']))

# Create model
model = SentimentModel(
    vocab_size=tokenizer.vocab_size,
    embed_size=CONFIG['embed_size'],
    hidden_size=CONFIG['hidden_size'],
    num_classes=CONFIG['num_classes'],
    num_layers=CONFIG['num_layers']
).to(CONFIG['device'])

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-3)
# scheduler = StepLR(optimizer, step_size=2, gamma=0.5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

train_model(model, train_loader, criterion, optimizer, scheduler, CONFIG['device'], CONFIG['num_epochs'])
evaluate_model(model, test_loader, CONFIG['device'])

torch.save(model.state_dict(), CONFIG['model_path'])