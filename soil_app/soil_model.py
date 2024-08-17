import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import accuracy_score

print("Script is running...")


# Define the CNN model
class SoilModel(nn.Module):
    def __init__(self, num_classes):
        super(SoilModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Hyperparameters
num_classes = 5  # Number of soil types
learning_rate = 0.001
batch_size = 32
epochs = 10

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load the datasets
train_dataset = datasets.ImageFolder(root='C:\\Users\\Sarang Pratham\\Desktop\\opencv\\Training', transform=transform)
test_dataset = datasets.ImageFolder(root='C:\\Users\\Sarang Pratham\\Desktop\\opencv\\Testing', transform=transform)

# Create DataLoader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model
model = SoilModel(num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
for epoch in range(epochs):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Testing the model
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

# Evaluate accuracy
accuracy = accuracy_score(all_labels, all_preds)
print(f"Accuracy: {accuracy}")

torch.save(model.state_dict(), 'soil_model.pth')