# 🐾 Cute Diffusion Model Project

This project is a modular implementation of a diffusion model for generating images. It's organized into a clean API structure to support easy use, training, and extension.

## ✨ Features

* Modular API for training and sampling
* Configurable parameters
* Easy to plug in your own dataset

## 🔧 Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/AAliAhmadi/Diffusion-Model.git
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## 🐱 Dataset Setup

To use the model with your custom dataset (e.g., cat images):

1. **Download the dataset archive**
   Download [`cat.rar`](#) (provide a real link here if hosted elsewhere).

2. **Extract the archive**
   Unzip or extract `cat.rar` to the project directory. After extraction, you should have or create a folder structure like:

   ```
   your_project/
   data/
   ├── cat/
   ├── cat/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   ```

3. **Use the dataset path**
   In your script or config, make sure to point to the correct dataset directory:

   ```python
   dataset_path = "./data/cat/"
   ```

## 🚀 Usage

Train the model:

```bash
python train.py --data-dir cat --epochs 100
```

Generate images:

```bash
python sample.py --output-dir samples/
```

## 🧠 How Diffusion Models Work (Short Summary)

Diffusion models are generative models that learn to reverse a gradual noising process. Starting from pure noise, the model learns to reconstruct data (e.g., images) by iteratively denoising it. It consists of:

1. **Forward Process**: Gradually add Gaussian noise to input data over many time steps.
2. **Reverse Process**: Train a neural network to predict and remove the noise at each step.
3. **Sampling**: Start with noise and apply the reverse process to generate realistic samples.

These models have been shown to produce high-quality, diverse images.

## 📁 Structure

```
├── data/            # Data loaders and preprocessing
├── model/           # Diffusion model components
├── train.py         # Training script
├── sample.py        # Sampling script
├── utils.py         # Utility functions
├── requirements.txt # Dependencies
├── README.md        # Project overview
```

## ❤️ Credits

Developed with love for learning and experimentation.
