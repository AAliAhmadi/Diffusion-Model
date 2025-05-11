import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from config import Config  # Import the Config class

def get_transforms():
    return transforms.Compose([
        transforms.Resize((Config.image_size, Config.image_size)),  # Use Config class here
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)  # Normalize the image to [-1, 1]
    ])

def get_dataloader():
    dataset = ImageFolder(root=Config.data_dir, transform=get_transforms())  # Use Config class here
    return DataLoader(dataset, batch_size=Config.batch_size, shuffle=True, drop_last=True)  # Use Config class here
