# SG-DDGAN: Semantic-Guided Dynamic Deblur GAN

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Status: Phase 1 (PoC)](https://img.shields.io/badge/Status-Phase_1_(PoC)-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)

> **Term Project for Computer Engineering, Abdullah Gul University**
> **Author:** Timur Yaşar

## 📖 Overview

The **Semantic-Guided Dynamic Deblur GAN (SG-DDGAN)** is a generative architecture designed to restore high-frequency details in dynamic video frames degraded by severe, spatially variant motion blur. 

Traditional 2D deblurring networks assume uniform blur degradation and fail to decouple global camera egomotion from local, independent object kinematics. This project bridges the gap between computationally heavy 3D continuous motion physics and real-time 2D inference by integrating a **Semantic Prior** to isolate dynamic foreground kinematics, with the eventual design goal of introducing a **Blur-Adaptive Neural Ordinary Differential Equation (ODE)** solver embedded in the latent space.

### 🔬 Current Project Status: Phase 1 (Proof of Concept)
This repository hosts the foundational **Phase 1: Micro-Training Architecture**, which has been benchmarked on a micro-split of the GOPRO dataset to validate structural stability and adversarial gradient flow. 

**Round 2 Architectural Upgrades:**
* **Active Semantic Injection:** The custom U-Net Generator has been upgraded to accept a 4-channel input (3 RGB channels + 1 Semantic Mask). This mathematically forces the latent space to process dynamic foreground objects independently.
* **Expanded Metrics:** Baseline comparative evaluation now includes Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM), and Learned Perceptual Image Patch Similarity (LPIPS).
* **Temporal Evaluation Pipeline:** A verified temporal Optical Flow (tOF) calculation pipeline has been implemented using a pre-trained RAFT model to measure multi-frame consistency.

## 🛠️ Requirements

The architecture requires a Python environment with PyTorch and standard computer vision libraries. 

```bash
pip install torch torchvision scikit-image opencv-python matplotlib lpips
```
🚀 Usage

To run the proof-of-concept training loop and verify the tOF metric pipeline:
Bash

python train.py

Expected Output:

  Semantic Generator Training: The script initializes the 4-channel Generator and PatchGAN Discriminator, running a 5-epoch micro-batch training session on dynamically generated tensors.

  Loss Graph: Outputs a convergence graph (loss_plot.png) demonstrating the Discriminator settling near 0.693 (maximum BCE uncertainty).

  RAFT Verification: The script seamlessly passes consecutive dummy frames through the pre-trained RAFT optical flow model to calculate and print the tOF Warping Error.

📂 Repository Structure
```
├── train.py           # Core implementation of the Semantic Generator, Discriminator, and tOF pipeline
├── loss_plot.png      # Output graph showing initial model convergence
└── README.md          # Project documentation
```

🔭 Future Work (Phase 2 Design Goals)

The current codebase establishes the baseline generative mechanics and active semantic routing. Future integration goals include:

   Neural ODE Solver (torchdiffeq): Integrating the mathematically formulated Dormand-Prince (dopri5) solver into the latent space to map continuous global egomotion trajectories using the Adjoint method.

  Full Dataset Execution: Transitioning from the micro-split constraints to multi-epoch training on the complete GoPro_Large and RealBlur datasets utilizing multi-GPU environments.

  Sequential tOF Benchmarking: Deploying the verified RAFT pipeline across complete video sequences to evaluate continuous temporal flickering reduction.

📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
