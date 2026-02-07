## Human Body Point Cloud Completion

A deep model for reconstructing complete human body point clouds from partial 3D scans. This implementation fine-tunes the Morphing and Sampling Network (MSN) architecture specifically for human body reconstruction, addressing occlusion and incomplete scanning challenges common in 3D acquisition systems.

## Overview

3D human body scanning systems commonly capture only partial geometry, especially when limited to a single frontal view. Acquiring complete body geometry typically requires multi-view scanning, subject rotation, or moving sensors, which increases system complexity and acquisition time. This work introduces a learning-based approach that reconstructs the back view of the human body from a single front-view scan, enabling full-body geometry generation without the need for multiple scans or dynamic scanners. Given a partial point cloud representing the frontal surface, the model infers anatomically plausible posterior geometry while preserving the observed structure.

The proposed method builds upon the Morphing and Sampling Network (MSN) framework introduced in [MSN-Point-Cloud-Completion](https://github.com/Colin97/MSN-Point-Cloud-Completion/tree/master). We adapt the architecture and training strategy specifically for human body reconstruction, emphasizing anatomical consistency and front-to-back shape inference rather than generic object completion. The resulting model generates dense, coherent full-body point clouds suitable for downstream applications such as avatar creation, virtual try-on, and human shape analysis.

## Key Features

Specialized for human body reconstruction – Fine-tuned on diverse human body datasets

End-to-end completion pipeline – Processes raw point clouds to complete reconstructions

Arbitrary partial inputs – Handles varying levels of incompleteness and occlusion

High-density output – Generates uniformly distributed point clouds suitable for downstream applications

Preservation of existing structure – Maintains accurate regions while completing missing parts

## Installation

### Repository Structure
```bash
├── MDS/                       # Multidimensional scaling utilities
├── expansion_penalty/         # Loss components for point distribution
├── results/                   # Output samples
├── dataset.py                 # Data loading and preprocessing
├── model.py                   # MSN architecture implementation
├── train.py                   # Training and fine-tuning script
├── halfpcd_to_completepcd.py  # Inference pipeline
├── utils.py                   # Helper functions and utilities
└── requirements.txt           # Python dependencies
```

### Prerequisites

Python 3.7+

CUDA 10.0 compatible GPU (recommended)

PyTorch 1.2.0

### Setup

```bash
# Clone repository
git clone https://github.com/imaryamsamani-lang/Body-Point-Cloud-Reconstruction.git
cd Body-Point-Cloud-Reconstruction

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Data Preparation

Organize your data with the following structure:

Store the main point clouds in "data/main" and partial point clouds in "data/partial"

```text

├── data/
│   ├── train/
│   │   ├── main/          # Ground-truth complete human body point clouds
│   │   └── partial/       # Frontal scans used as input
│   └── test/
│       ├── main/
│       └── partial/

```

Supported formats: .ply, .xyz, .npy, .pcd

### Model Weights

Download pretrained weights for human body completion:  [halfpcd_to_completepcd.pth](https://drive.google.com/file/d/1FVso6CyGykl2pQbWLBvpL0wOG61xStcO/view?usp=sharing)

Place the downloaded file in the project root directory.

### Visualize

Visualize complete the reconstructed point clouds:

```bash
python halfpcd_to_completepcd.py
```

### Training and Fine-tuning

To train on new datasets or fine-tune the model:

```bash
python train.py
```

Modify training parameters within train.py as needed:

Batch size and learning rate

Data augmentation strategies

Loss function weights

Training epochs and validation frequency

## Technical Details

### Model Architecture

The MSN framework employs a two-stage approach:

Morphing: Deforms a set of simple primitives to match the input structure

Sampling: Generates dense, uniformly distributed points from the morphed representation

This dual process ensures both structural accuracy and point distribution quality in the completed output.

### Training Configuration

Input: 2,048 points (partial point cloud)

Output: 16,384 points (completed point cloud)

Loss: Combined Chamfer distance and expansion penalty

Optimizer: Adam with learning rate scheduling

## Results

Example reconstructions demonstrate the model's capability to generate complete human body point clouds from severely occluded inputs while maintaining anatomical plausibility.

![Diagram](results/sample.png)
