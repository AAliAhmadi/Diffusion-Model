import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from config import IMG_SIZE, DATA_PATH, BATCH_SIZE

def get_transforms():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])

def get_dataloader():
    dataset = ImageFolder(root=DATA_PATH, transform=get_transforms())
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)