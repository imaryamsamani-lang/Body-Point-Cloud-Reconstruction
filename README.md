## Human Body Point Cloud Completion

This repository is a fine‑tuned point cloud completion model designed specifically for reconstructing human body point clouds. Given an incomplete 3D point cloud as input, the model generates and outputs a complete point cloud.

The implementation is based on the Morphing and Sampling Network (MSN) — a learning‑based dense point cloud completion framework from the original repository: [MSN-Point-Cloud-Completion](https://github.com/Colin97/MSN-Point-Cloud-Completion/tree/master)

## 🔍 Overview

Most real‑world 3D scanning systems (e.g., depth sensors or LiDAR) produce incomplete point clouds due to occlusion, limited viewpoints, or sensor noise. Point cloud completion tackles this by generating a dense and complete 3D shape from partial observations.

In this project:

You’ll find a version of MSN fine‑tuned on human body point clouds.

Given a partial scan of a human body, the model predicts a complete, high‑quality reconstruction.

The original MSN method preserves known structures and generates dense, uniformly distributed point clouds using a morphing‑and‑sampling strategy.

## 🚀 Features

Fine‑tuned for human body shapes

End‑to‑end deep learning model for point cloud completion

Works with arbitrary incomplete inputs

Produces dense and evenly distributed output point clouds

## 📁 Repository Structure
```bash
├── MDS/
├── expansion_penalty/
├── README.md
├── dataset.py
├── halfpcd_to_completepcd.py
├── model.py
├── train.py
└── utils.py
```

## 🧠 Setup & Dependencies

Install Python dependencies:

Make sure you have:

Pytorch 1.2.0

CUDA 10.0

Python 3.7

Visdom

Open3D

```bash
git clone https://github.com/imaryamsamani-lang/Body-Point-Cloud-Reconstruction.git
cd human‑body‑pc‑completion
```

```bash
pip install ‑r requirements.txt
```

## 📦 Using the Model
1. Prepare Data

Store the main point clouds in "data/main" and partial point clouds in "data/partial"

Ensure files are in .ply, .xyz, or supported point cloud format.

2. Inference (Completion)

Download the weights at: [halfpcd_to_completepcd.pth](https://drive.google.com/file/d/1FVso6CyGykl2pQbWLBvpL0wOG61xStcO/view?usp=sharing)

```bash
python halfpcd_to_completepcd.py 
```

Run the validation/completion script:

The script will load the fine‑tuned model and generate complete point clouds under outputs/completion_results.

3. Training / Fine‑Tuning (Optional)

To further fine‑tune on new human body datasets:
```bash
python train.py 
```

Adjust parameters in the script (learning rate, batch size, data augmentations) as needed.

## 📊 Sample Outputs

Sample outputs:
![Diagram](results/sample.png)
