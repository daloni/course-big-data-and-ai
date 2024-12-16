import os
import random
from torchvision import transforms
from PIL import Image
import torch
from NeuronalNetwork import NeuronalNetwork

# Carpetas con las imágenes
benign_dir = "./data/melanoma_cancer_dataset/test/benign"
malignant_dir = "./data/melanoma_cancer_dataset/test/malignant"

transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def load_random_images(folder, num_images):
    all_images = os.listdir(folder)
    selected_images = random.sample(all_images, num_images)
    loaded_images = []
    for img_name in selected_images:
        img_path = os.path.join(folder, img_name)
        image = Image.open(img_path).convert('RGB')  # Convertir a RGB si no lo está
        loaded_images.append((img_name, transform(image)))
    return loaded_images

benign_images = load_random_images(benign_dir, 5)
malignant_images = load_random_images(malignant_dir, 5)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuronalNetwork().to(device)
model.load_state_dict(torch.load("melanoma_cnn.pth"))
model.eval()

print("Benign")
for img_name, img_tensor in benign_images:
    img_tensor = img_tensor.unsqueeze(0).to(device) 
    output = model(img_tensor)
    _, predicted = torch.max(output, 1)
    class_name = "benign" if predicted.item() == 0 else "malignant"
    print(f"Imagen: {img_name}, Predicción: {class_name}")

print("Malignant")
for img_name, img_tensor in malignant_images:
    img_tensor = img_tensor.unsqueeze(0).to(device) 
    output = model(img_tensor)
    _, predicted = torch.max(output, 1)
    class_name = "benign" if predicted.item() == 0 else "malignant"
    print(f"Imagen: {img_name}, Predicción: {class_name}")
