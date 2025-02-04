import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import numpy as np


# Set random seed for reproducibility
torch.manual_seed(42)


# Transform images to tensors and normalize
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


# Load EMNIST dataset with byclass split (includes uppercase, lowercase, and digits)
train_dataset = datasets.EMNIST(root='./data', split='byclass', train=True, transform=transform, download=True)
test_dataset = datasets.EMNIST(root='./data', split='byclass', train=False, transform=transform, download=True)


# Create DataLoaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


# Define the CNN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 62)  # 62 classes: 10 digits + 26 uppercase + 26 lowercase


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


# Initialize model, optimizer, and loss function
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


# Training loop
print("Starting training...")
for epoch in range(5):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
       
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


# Evaluation
print("\nEvaluating model...")
model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += criterion(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()


test_loss /= len(test_loader.dataset)
print(f'Test set: Average loss: {test_loss:.4f}, '
      f'Accuracy: {correct}/{len(test_loader.dataset)} '
      f'({100. * correct / len(test_loader.dataset):.2f}%)\n')


# New save path
save_path = 'D:/ESAD/Travail autonomie/A quoi revent les algorythmes/alphabet2'
os.makedirs(save_path, exist_ok=True)


# Function to convert EMNIST label to character
def get_character(label):
    if label < 10:  # Digits
        return f'digit_{label}'
    elif label < 36:  # Uppercase
        return f'upper_{chr(label - 10 + 65)}'
    else:  # Lowercase
        return f'lower_{chr(label - 36 + 97)}'


# Generate and save images
print("Generating character images...")
model.eval()


# Create a figure for combined visualization
num_rows = 8
num_cols = 8
fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 20))
axes = axes.ravel()


with torch.no_grad():
    # Get one example of each class
    for idx in range(62):  # 62 classes total
        # Find an example of the current class
        for data, target in test_loader:
            mask = target == idx
            if mask.any():
                sample = data[mask][0].to(device)
                break
       
        # Get the image and normalize it for saving
        img = sample.cpu().numpy()[0]
        img = ((img - img.min()) * 255 / (img.max() - img.min())).astype(np.uint8)
       
        # Save individual image
        char = get_character(idx)
        plt.imsave(os.path.join(save_path, f'{char}.png'), img, cmap='gray')
       
        # Add to combined visualization
        axes[idx].imshow(img, cmap='gray')
        axes[idx].axis('off')
        axes[idx].set_title(f'{char}')


# Hide empty subplots
for idx in range(62, len(axes)):
    axes[idx].axis('off')


# Save combined visualization
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'combined_characters.png'))
plt.close()


print(f"Generated images have been saved to: {save_path}")