from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# from NeuronalNetwork import NeuronalNetwork
from NeuronalNetworkResnet import NeuronalNetwork

transform = transforms.Compose([
    transforms.Resize((200, 200)), # Redimension images
    transforms.ToTensor(), # Convert to tensor
    transforms.Normalize([0.5], [0.5]) # Normalize
])

# data_dir = "./data/DermMel"
data_dir = "./data/melanoma_cancer_dataset"

train_dataset = datasets.ImageFolder(f"{data_dir}/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = datasets.ImageFolder(f"{data_dir}/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Use device:', device)
model = NeuronalNetwork().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

# Bucle de entrenamiento
def train(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs=10):
    for epoch in range(epochs):
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

        scheduler.step()

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

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(test_loader):.4f}, Val Accuracy: {100 * correct / total:.2f}%")

train(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs=5)

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

print(classification_report(y_true, y_pred, target_names=train_dataset.classes))

torch.save(model.state_dict(), "melanoma_cnn.pth")
