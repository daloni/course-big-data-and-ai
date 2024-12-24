from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from NeuronalNetworkResnet import NeuronalNetwork
import time

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_dir = "../data/spectograms-train"

dataset = datasets.ImageFolder(f"{data_dir}", transform=transform)

# Split the dataset into train (80%) and test (20%)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Use device:', device)
model = NeuronalNetwork().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
# scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1, min_lr=1e-6)

def train(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs=10):
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        finish_time = time.time() - start_time

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(test_loader)
        train_loss /= len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{epochs} completed in {finish_time:.2f} seconds, "
              f"Learning Rate: {current_lr:.6f}, "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Accuracy: {100 * correct / total:.2f}%")

train(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs=15)

from sklearn.metrics import classification_report

model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

print(classification_report(y_true, y_pred, target_names=dataset.classes))

torch.save(model.state_dict(), "model.pth")
