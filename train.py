import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import Config
from dataset import CustomDataset
from model import UNet
from scheduler import DiffusionScheduler
from utils import get_device, set_seed, save_images


def train():
    config = Config()
    set_seed(config.seed)
    device = get_device()

    # Dataset
    dataset = CustomDataset(config.data_dir, image_size=config.image_size)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)

    # Model and scheduler
    model = UNet().to(device)
    scheduler = DiffusionScheduler(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    model.train()
    for epoch in range(config.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}")
        for images in pbar:
            images = images.to(device)
            t = torch.randint(0, config.timesteps, (images.size(0),), device=device).long()
            noise = torch.randn_like(images)
            x_t = scheduler.q_sample(images, t, noise=noise)

            predicted_noise = model(x_t, t)
            loss = torch.nn.functional.mse_loss(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item())

        # Optional: Save sample
        with torch.no_grad():
            sampled = scheduler.sample(model, (8, 3, config.image_size, config.image_size), device)
            save_images(sampled, f"samples/sample_epoch_{epoch+1}.png")


if __name__ == "__main__":
    train()
