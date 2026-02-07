## Human Body Point Cloud Completion

A deep model for reconstructing complete human body point clouds from partial 3D scans. This implementation fine-tunes the Morphing and Sampling Network (MSN) architecture specifically for human body reconstruction, addressing occlusion and incomplete scanning challenges common in 3D acquisition systems.

## Overview

3D scanning systems such as depth sensors and LiDAR often produce incomplete point clouds due to occlusion, limited viewpoints, and sensor noise. This repository provides a specialized solution for human body completion, generating dense, high-quality reconstructions from partial inputs. The model preserves existing structural details while generating plausible completions for missing regions.

The core methodology builds upon the Morphing and Sampling Network (MSN) framework from the original work [MSN-Point-Cloud-Completion](https://github.com/Colin97/MSN-Point-Cloud-Completion/tree/master). Our adaptation focuses on human body shapes, with modifications to architecture and training protocols optimized for anatomical structures.

## Key Features

Specialized for human body reconstruction – Fine-tuned on diverse human body datasets

End-to-end completion pipeline – Processes raw point clouds to complete reconstructions

Arbitrary partial inputs – Handles varying levels of incompleteness and occlusion

High-density output – Generates uniformly distributed point clouds suitable for downstream applications

Preservation of existing structure – Maintains accurate regions while completing missing parts

## Installation

### Repository Structure
```bash
├── MDS/                    # Multidimensional scaling utilities
├── expansion_penalty/      # Loss components for point distribution
├── dataset.py              # Data loading and preprocessing
├── model.py                # MSN architecture implementation
├── train.py                # Training and fine-tuning script
├── halfpcd_to_completepcd.py  # Inference pipeline
├── utils.py                # Helper functions and utilities
└── requirements.txt        # Python dependencies
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
data/
├── main/          # Complete point clouds (ground truth)
└── partial/       # Corresponding partial point clouds
```

Supported formats: .ply, .xyz, .npy, .pcd

### Model Weights
Download pretrained weights for human body completion:  [halfpcd_to_completepcd.pth](https://drive.google.com/file/d/1FVso6CyGykl2pQbWLBvpL0wOG61xStcO/view?usp=sharing)

Place the downloaded file in the project root directory.

### Inference
Generate complete point clouds from partial inputs:

```bash
python halfpcd_to_completepcd.py
```

Outputs will be saved to outputs/completion_results/ with corresponding filename identifiers.

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
