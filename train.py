#!/usr/bin/env python
# coding: utf-8

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import Config
from dataset import get_dataloader  # Use get_dataloader function for loading dataset
from model import UNet
from scheduler import DiffusionScheduler
from utils import get_device, set_seed, save_images

def train():
    # Set up configuration and device
    config = Config()
    set_seed(config.seed)
    device = get_device()

    # Initialize DataLoader
    dataloader = get_dataloader()

    # Model and scheduler setup
    model = UNet().to(device)
    scheduler = DiffusionScheduler(
        timesteps=config.timesteps,
        start_beta=config.beta_start,
        end_beta=config.beta_end
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Training loop
    model.train()
    for epoch in range(config.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}")
        for batch in pbar:
            images = batch[0]  # Assuming the dataset returns (image, label)
            images = images.to(device)

            t = torch.randint(0, config.timesteps, (images.size(0),), device=device).long()
            noise = torch.randn_like(images)
            x_t = scheduler.q_sample(images, t, noise=noise)

            predicted_noise = model(x_t, t)
            loss = torch.nn.functional.mse_loss(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update progress bar with loss
            pbar.set_postfix(loss=loss.item())

        # Optional: Save sample every epoch
        with torch.no_grad():
            sampled = scheduler.sample(model, (8, 3, config.image_size, config.image_size), device)
            save_images(sampled, f"samples/sample_epoch_{epoch+1}.png")

if __name__ == "__main__":
    train()
