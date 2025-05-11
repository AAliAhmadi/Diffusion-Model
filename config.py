# config.py
import torch

class Config:
    # Dataset and image parameters
    image_size = 64  # Size of the images (for example, 64x64)
    channels = 3  # Number of channels in images (3 for RGB, 1 for grayscale)
    batch_size = 64  # Number of images per batch

    # Training parameters
    epochs = 100  # Number of training epochs
    lr = 2e-4  # Initial learning rate for the optimizer
    beta_start = 1e-4  # Starting value of beta (used in the diffusion process)
    beta_end = 0.02  # Ending value of beta (used in the diffusion process)
    timesteps = 1000  # Number of diffusion timesteps

    # Dataset path
    data_dir = "./data/cat/"  # Path to your dataset folder (replace with your folder name or path)

    # Checkpoints and logs
    save_dir = "checkpoints"  # Directory to save model checkpoints
    log_dir = "logs"  # Directory for logs (e.g., TensorBoard logs)

    # Device setting (automatic GPU/CPU detection)
    device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available, else CPU

    # Seed for reproducibility
    seed = 42  # Random seed for reproducibility
