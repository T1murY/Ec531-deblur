# SG-DDGAN: Semantic-Guided Dynamic Deblur GAN

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Status: Proof of Concept](https://img.shields.io/badge/Status-Phase_1_(PoC)-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)

> **Term Project for Computer Engineering, Abdullah Gul University**
> **Author:** Timur Yaşar

## 📖 Overview

The **Semantic-Guided Dynamic Deblur GAN (SG-DDGAN)** is a generative architecture designed to restore high-frequency details in dynamic video frames degraded by severe, spatially variant motion blur. 

Traditional 2D deblurring networks assume uniform blur degradation and fail to decouple global camera egomotion from local, independent object kinematics. This project bridges the gap between computationally heavy 3D continuous motion physics (e.g., NeRFs/3DGS) and real-time 2D inference by integrating a **Semantic Prior** with a **Blur-Adaptive Neural Ordinary Differential Equation (ODE)** solver embedded in the latent space of a Generative Adversarial Network.

### 🔬 Current Project Status: Phase 1 (Proof of Concept)
This repository currently hosts the foundational **Phase 1: Micro-Training Architecture**. To validate structural stability and adversarial gradient flow before introducing computationally heavy Neural ODEs, a lightweight U-Net Generator and PatchGAN Discriminator have been implemented from scratch.

**Current capabilities in this repository:**
* Custom PyTorch `SimpleGenerator` (U-Net topology).
* Custom PyTorch `SimpleDiscriminator` (PatchGAN topology).
* A self-contained micro-training loop confirming adversarial and $L_1$ pixel loss convergence.

## 🛠️ Requirements

The base architecture requires a Python environment with PyTorch. 

```bash
pip install torch torchvision numpy matplotlib opencv-python
