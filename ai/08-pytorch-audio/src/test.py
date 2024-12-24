import os
import random
from torchvision import transforms
from PIL import Image
import torch
from NeuronalNetworkResnet import NeuronalNetwork
from target_relation import TARGET

ONLY_ONE_FOLDER = True
RANDOM_IMAGES = 20

# Carpetas con las imágenes
folder = "../data/spectograms-train"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

real_all_folders = os.listdir(folder)
real_all_folders.sort()

all_folders = real_all_folders
all_folder_images = []

if ONLY_ONE_FOLDER:
    all_folders = [all_folders[random.randint(0, len(all_folders) - 1)]]

for folder_name in all_folders:
    all_folder_images.append((folder_name, load_random_images(os.path.join(folder, folder_name), RANDOM_IMAGES)))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuronalNetwork().to(device)
model.load_state_dict(torch.load("model.pth", weights_only=False))
model.eval()

for folder_name, images in all_folder_images:
    print(f"Folder: {folder_name}")
    for img_name, img_tensor in images:
        img_tensor = img_tensor.unsqueeze(0).to(device) 
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        print(f"Image: {img_name}, Predicted: {real_all_folders[predicted.item()]}")
    print()
