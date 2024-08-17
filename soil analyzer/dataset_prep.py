import torch
from torchvision import transforms, datasets

# Define transformations (you can customize these based on your needs)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Create a dataset
dataset = datasets.ImageFolder(root='Soil_types', transform=transform)


# Create a data loader
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Example: Iterate over batches
for inputs, labels in data_loader:
    # Your training or testing logic here
    print(inputs.shape, labels)
