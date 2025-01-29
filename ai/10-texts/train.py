import torch
import torch.optim as optim
import pandas as pd
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, BertForSequenceClassification, get_scheduler
from tqdm import tqdm

from ReviewDataset import ReviewDataset
from config import CONFIG

def train_model(model, train_loader, criterion, optimizer, scheduler, device, num_epochs=5):
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        optimizer.zero_grad()

        for step, data in enumerate(progress_bar):
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask).logits
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            progress_bar.set_postfix(loss=total_loss/len(train_loader), accuracy=100*correct/total)

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc=f"Evaluating")

        for data in progress_bar:
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask).logits
            predictions = torch.argmax(outputs, dim=1)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(classification_report(all_labels, all_preds, target_names=['Negative', 'Positive']))

df = pd.read_csv(CONFIG['dataset_path'])
reviews = df['review'].values
labels = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0).values

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
X_train, X_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.2, random_state=42)

train_dataset = ReviewDataset(X_train, y_train, tokenizer, CONFIG['max_len'])
test_dataset = ReviewDataset(X_test, y_test, tokenizer, CONFIG['max_len'])

train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

# Create model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(CONFIG['device'])

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-3)

scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * CONFIG['num_epochs'])

train_model(model, train_loader, criterion, optimizer, scheduler, CONFIG['device'], CONFIG['num_epochs'])
evaluate_model(model, test_loader, CONFIG['device'])

torch.save(model.state_dict(), CONFIG['model_path'])
