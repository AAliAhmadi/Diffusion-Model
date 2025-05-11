import os
import torch
import numpy as np
from torchvision.utils import save_image

def save_images(images, path, nrow=8):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_image(images, path, nrow=nrow)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    import random
    import torch
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
