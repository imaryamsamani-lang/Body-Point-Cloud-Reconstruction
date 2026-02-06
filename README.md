## Human Body Point Cloud Completion

This repository contains a fine‑tuned point cloud completion model designed specifically for incomplete human body point clouds. Given an incomplete 3D point cloud as input, the model reconstructs and outputs a complete point cloud.

The implementation is based on the Morphing and Sampling Network (MSN) — a learning‑based dense point cloud completion framework from the original repository: https://github.com/Colin97/MSN-Point-Cloud-Completion/tree/master

🔍 Overview

Most real‑world 3D scanning systems (e.g., depth sensors or LiDAR) produce incomplete point clouds due to occlusion, limited viewpoints, or sensor noise. Point cloud completion tackles this by generating a dense and complete 3D shape from partial observations.

In this project:

You’ll find a version of MSN fine‑tuned on human body point clouds.

Given a partial scan of a human body, the model predicts a complete, high‑quality reconstruction.

The original MSN method preserves known structures and generates dense, uniformly distributed point clouds using a morphing‑and‑sampling strategy.

🚀 Features

Fine‑tuned for human body shapes

End‑to‑end deep learning model for point cloud completion

Works with arbitrary incomplete inputs

Produces dense and evenly distributed output point clouds

📁 Repository Structure
├── MDS/
├── expansion_penalty/
├── README.md
├── dataset.py
├── halfpcd_to_completepcd.py
├── model.py
├── train.py
└── utils.py

🧠 Setup & Dependencies

Install Python dependencies:

```bash
git clone https://github.com/yourusername/human‑body‑pc‑completion.git
cd human‑body‑pc‑completion
```

# Create and activate a virtual environment
```bash
python3 ‑m venv venv
source venv/bin/activate
```

# Install requirements
```bash
pip install ‑r requirements.txt
```

Make sure you have:

Python 3.7+

PyTorch (compatible with your GPU)

Open3D (for visualization)

Optional: CUDA support for GPU acceleration

📦 Using the Model
1. Prepare Data

Ensure files are in .ply, .xyz, or supported point cloud format.

2. Inference (Completion)

```bash
python halfpcd_to_completepcd.py \
    --model_path weights/halfpcd_to_completepcd.pth \
    --input_dir data/partial_pointclouds \
    --output_dir data/completed_pointclouds
```

Run the validation/completion script:

The script will load the fine‑tuned model and generate complete point clouds under outputs/completion_results.

3. Training / Fine‑Tuning (Optional)

To further fine‑tune on new human body datasets:
```bash
python train.py \
    --data_dir data/human_body \
    --save_dir models/ \
    --epochs 100 \
    --batch_size 16
```

Adjust parameters in the script (learning rate, batch size, data augmentations) as needed.

📊 Evaluation

Use standard metrics such as:

Chamfer Distance (CD)

Earth Mover’s Distance (EMD)

These help quantify the similarity between the predicted and ground truth point clouds.

🧠 Acknowledgements

This project leverages the MSN framework for dense point cloud completion, adapting it to the domain of human body reconstruction.
