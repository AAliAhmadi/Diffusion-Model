
# 🐍 Diffusion Model Playground 

Welcome to the **Diffusion Model Modular API** – a cuddly and clean PyTorch-based implementation of a Denoising Diffusion Probabilistic Model (DDPM)! This repo is designed with modularity and clarity in mind, making it perfect for learning, experimenting, and extending 🧪✨.

---

## 🧠 What is a Diffusion Model?

Diffusion models are a class of generative models that learn to create new data by reversing a noising process. Here's how it works in simple terms:

1. **Forward Process (Noise In)** 🌫️: Gradually add Gaussian noise to a real image over many steps until it becomes pure noise.
2. **Reverse Process (Denoise Out)** 🌈: Train a neural network to reverse this process step-by-step, turning the noise back into a beautiful image!

These models are popular because they can generate **high-quality, diverse images**, and have been used in state-of-the-art tools like **DALL·E 2** and **Stable Diffusion**.

---

## 🧩 Project Structure

```
diffusion_model_modular/
├── configs/
│   └── default_config.py      # 🛠️ Configuration settings (image size, channels, timesteps, etc.)
├── models/
│   └── unet.py                # 🧠 A simple UNet architecture used for denoising
├── diffusion/
│   └── gaussian_diffusion.py # 🔁 The core diffusion logic (forward + reverse processes)
├── utils/
│   └── utils.py               # 🧰 Helper functions (e.g., beta schedules, image saving)
├── main.py                    # 🚀 Training entry point
└── README.md                  # 📖 You're reading it!
```

---

## 🚀 Getting Started

1. Clone the repo:

```bash
git clone https://github.com/AAliAhmadi/diffusion-model-modular.git
cd diffusion-model-modular
```

2. Install requirements:

```bash
pip install -r requirements.txt
```

3. Train your model:

```bash
python main.py
```

You can change parameters in `configs/default_config.py` to adjust image size, diffusion steps, and more.

---

## 🖼️ Output Samples

After training, generated samples will be saved to the `samples/` directory. Enjoy watching your model create magic from noise! ✨🎨

---

## 💌 Credits

This project was built with love using PyTorch 🐍❤️ and inspired by the fantastic work in the diffusion model research community.

---

## 📬 Questions or Ideas?

Feel free to open issues or reach out with suggestions! Let's make this project even more awesome together 🧸🌟.

Happy Generating! 🐰🎈
