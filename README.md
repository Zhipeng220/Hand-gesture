# Sign Language Recognition System - Based on Anatomical Constraints and Spatial Attention Mechanisms

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Table of Contents

- [Introduction](#introduction)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Algorithm Description](#algorithm-description)
- [Experiment Reproduction](#experiment-reproduction)
- [Model Evaluation](#model-evaluation)
- [Embedded Deployment](#embedded-deployment)
- [Experimental Results](#experimental-results)
- [FAQ](#faq)
- [Citation](#citation)

---

## 🎯 Introduction

> **Note:** This repository is the official code implementation for our paper submitted to **The Visual Computer**:
>
> **"Enhancing Static Sign Language Recognition: A Spatial Attention and Edge Computing Approach"**
>
> If you use this code or dataset, please cite our work (see [Citation](#citation) section).

This project aims to build a high-accuracy, low-latency sign language recognition system...
### Main Contributions

- 🔬 **First-of-its-kind Integration**: First end-to-end fusion of anatomical constraints with deep learning models
- 📈 **Significant Performance Improvement**: Average recognition accuracy of 99.86%, 12.6 percentage points higher than traditional CNN methods
- ⚡ **Real-time Inference**: Achieves 32FPS real-time inference on Jetson Xavier NX
- 🎯 **Strong Robustness**: Maintains high accuracy under complex backgrounds

---

## ✨ Key Features

### 1. Anatomical Constraints + Spatial Attention Mechanism
- Introduces **joint kinematic constraints** and **phalangeal length ratio constraints**
- Spatial attention mechanism based on MultiheadAttention optimizes feature weight distribution
- Ensures biological plausibility and improves recognition performance in complex scenes

### 2. Data Augmentation Strategy
- Uses SMOTE oversampling to address class imbalance issues
- Image augmentation (rotation, flipping, translation, noise) expands training data
- Intelligent filtering mechanism removes invalid augmented samples

### 3. Embedded Platform Optimization
- Model size compressed by 68%
- TensorRT accelerated inference
- Achieves 32FPS real-time performance

---

## 🏗️ System Architecture

![System Architecture Diagram](./images/architecture.png)
---

## 🔧 Environment Setup

### Hardware Requirements

#### Training Environment
- GPU: NVIDIA GTX 1060 or above (6GB+ VRAM)
- CPU: Intel i5 or above
- RAM: 16GB+
- Storage: 10GB+ available space

#### Inference Environment (Embedded)
- Jetson Xavier NX / Jetson Nano
- CUDA 11.4 / cuDNN 8.2
- TensorRT 8.2

### Software Dependencies

#### Basic Environment
```bash
Python 3.8+
CUDA 11.4+ (for GPU training)
cuDNN 8.2+
```

#### Python Package Dependencies

Create virtual environment:
```bash
conda create -n sign_language python=3.8
conda activate sign_language
```

Install dependencies:
```bash
pip install -r requirements.txt
```

**requirements.txt content:**
```txt
# Deep Learning Framework
torch>=1.10.0
torchvision>=0.11.0

# Data Processing
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0
imbalanced-learn>=0.8.0

# Computer Vision
opencv-python>=4.5.0
mediapipe>=0.8.9

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0

# Model Saving
joblib>=1.0.0

# Other Tools
tqdm>=4.62.0
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
python -c "import mediapipe; print('MediaPipe installed successfully')"
```

---

## 📁 Data Preparation

### Directory Structure

```
HandFind/
├── data/                          # Original data directory
│   ├── video/                     # Original video files
│   │   ├── hello_001.mp4
│   │   ├── thanks_001.mp4
│   │   └── ...
│   └── picture/                   # Original frames extracted from videos
│
├── data1/                         # Processed data directory
│   ├── picture_augmented/         # Augmented images
│   ├── find_hand.csv             # Hand keypoint coordinates
│   ├── find.csv                  # Cleaned coordinate data
│   ├── change.csv                # Final data with reconstructed labels
│   └── log.txt                   # Processing logs
│
├── src/                          # Source code directory
│   ├── VideoCreatePicture.py     # Video to image conversion
│   ├── DataPictureAugmented.py   # Data augmentation
│   ├── FindHandCsv.py            # Extract hand keypoints
│   ├── FindImgnameError.py       # Handle recognition errors
│   ├── DelAgmentedPicture.py     # Delete invalid augmented images
│   ├── Clean.py                  # Data cleaning
│   ├── TrainModel.py             # Model training
│   └── Pridict.py                # Model prediction
│
├── models/                       # Model files
│   ├── best_model.pth            # Best model weights
│   ├── scaler.pkl                # Data scaler
│   └── label_encoder.pkl         # Label encoder
│
├── saved_images/                 # Prediction results
│   ├── confusion_matrix.png
│   ├── metrics_summary.png
│   └── ...
│
├── requirements.txt              # Dependency list
└── README.md                     # Project documentation
```

### Dataset Description

This project supports custom sign language datasets with the following requirements:

1. **Video Format**: MP4, AVI, MOV, MKV
2. **Naming Convention**: `{gesture_class}_{number}.{extension}`
   - Example: `hello_001.mp4`, `thanks_002.mp4`
3. **Video Requirements**:
   - Resolution: 640×480 or higher
   - Frame rate: 30fps or higher
   - Hand clearly visible, occupying at least 30% of the frame

---

## 🧠 Algorithm Description

### 1. Hand Keypoint Detection

Uses **MediaPipe Hands** for hand keypoint detection:

```python
# MediaPipe detects 21 hand keypoints
# Each keypoint contains (x, y, z) 3D coordinates
# Supports two-hand detection, total 42 keypoints (21×2)

Keypoint numbering:
0: Wrist (WRIST)
1-4: Thumb (THUMB_CMC, MCP, IP, TIP)
5-8: Index Finger (INDEX_FINGER_MCP, PIP, DIP, TIP)
9-12: Middle Finger (MIDDLE_FINGER_MCP, PIP, DIP, TIP)
13-16: Ring Finger (RING_FINGER_MCP, PIP, DIP, TIP)
17-20: Pinky (PINKY_MCP, PIP, DIP, TIP)
```

**Key Code Implementation:**
```python
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)

# Detect hands
results = hands.process(rgb_image)

# Extract keypoint coordinates
for landmarks in results.multi_hand_landmarks:
    coords = [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]
```

### 2. Spatial Attention Model Architecture

**Model Components:**

1. **Coordinate Embedding Layer**
   - Maps 3D coordinates (x,y,z) to 64-dimensional feature space
   - Enhances model's ability to represent spatial positions

2. **Multi-Head Attention Mechanism**
   - 8 attention heads computing in parallel
   - Captures spatial relationships between different keypoints
   - Adaptively learns importance weights of keypoints

3. **Residual Connection**
   - Alleviates gradient vanishing problem
   - Preserves original position information

4. **Feature Fusion**
   - Flattens all keypoint features
   - Fully connected layer for classification

**Core Code:**
```python
class StaticGestureAttnModel(nn.Module):
    def __init__(self, input_dim=3, num_classes=10, num_heads=8):
        super().__init__()
        self.embed_dim = 64
        
        # Coordinate embedding
        self.coord_embed = nn.Linear(input_dim, self.embed_dim)
        
        # Multi-head attention
        self.attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(self.embed_dim * 42, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # x: [batch, 42, 3]
        embedded = self.coord_embed(x)  # [batch, 42, 64]
        attn_out, weights = self.attn(embedded, embedded, embedded)
        fused = embedded + attn_out  # Residual connection
        return self.fc(fused.view(fused.size(0), -1))
```

### 3. Data Augmentation Strategy

**Image Augmentation Methods:**

| Augmentation Type | Parameter Range | Purpose |
|-------------------|----------------|---------|
| Rotation | ±45° | Adapt to different gesture angles |
| Flip | Horizontal flip | Increase left/right hand samples |
| Translation | ±20% | Adapt to different hand positions |
| Noise | σ=25 | Improve noise resistance |

**SMOTE Oversampling:**
- Addresses class imbalance issues
- Synthesizes minority class samples in feature space
- Sampling strategy: Automatically balances sample counts across classes

**Key Code:**
```python
from imblearn.over_sampling import SMOTE

# SMOTE oversampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, encoded_labels)
```

### 4. Training Strategy

**Optimizer Configuration:**
```python
optimizer = Adam(
    model.parameters(),
    lr=1e-5,              # Learning rate
    weight_decay=1e-4     # L2 regularization
)

# Learning rate scheduler
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='max',           # Monitor validation accuracy
    factor=0.5,           # Decay factor
    patience=3            # Patience epochs
)
```

**Early Stopping Mechanism:**
- Monitors validation set accuracy
- Stops training if no improvement for 5 consecutive epochs
- Automatically saves best model

---

## 🚀 Experiment Reproduction

### Complete Workflow

#### Step 1: Clone Project

```bash
git clone https://github.com/Scatteredpeople/HandFind.git
cd HandFind
```

#### Step 2: Download and Prepare Data

The raw video dataset used for this project is hosted on Zenodo.

1.  **Download the dataset:**
    * **DOI:** [https://doi.org/10.5281/zenodo.17708961](https://doi.org/10.5281/zenodo.17708961)
    * Download the video files from the Zenodo link above.

2.  **Place the videos:**
    * Create the video directory:
        ```bash
        mkdir -p data/video
        ```
    * Move all downloaded `.mp4` video files into the `data/video/` directory.

#### Step 3: Video to Image Conversion

```bash
cd src
python VideoCreatePicture.py
```

**Parameter Description:**
- `video_folder`: Video input path (default: `./data/video`)
- `output_folder`: Image output path (default: `./data/picture`)
- `fps_target`: Extraction frame rate (default: 30fps)

**Output:** Frame images generated in `data/picture/` directory

#### Step 4: Data Augmentation

```bash
python DataPictureAugmented.py
```

**Parameter Description:**
- `input_dir`: Original image path
- `output_dir`: Augmented image output path
- `num_augmentations`: Number of augmented samples per image (default: 5)

**Output:** Augmented images generated in `data1/picture_augmented/` directory

#### Step 5: Extract Hand Keypoints

```bash
python FindHandCsv.py
```

**Processing Flow:**
1. Use MediaPipe to detect hands in each image
2. Extract coordinates of 42 keypoints (21 points per hand for two hands)
3. Save as CSV file

**Output:** `data1/find_hand.csv`

#### Step 6: Data Cleaning

```bash
# Handle failed detection images
python FindImgnameError.py

# Delete invalid augmented images
python DelAgmentedPicture.py

# Clean data and reconstruct labels
python Clean.py
```

**Cleaning Rules:**
- Delete samples where both hand coordinates are 0 or NaN
- Extract label prefix from filename as class label
- Delete corresponding invalid augmented images

**Output:** `data1/change.csv` (final training data)

#### Step 7: Model Training

```bash
python TrainModel.py
```

**Training Configuration:**
```python
epochs = 100                # Maximum training epochs
batch_size = 16            # Batch size
learning_rate = 1e-5       # Initial learning rate
early_stop_patience = 5    # Early stopping patience
```

**Training Process Monitoring:**
- Output per epoch: Training loss, validation accuracy
- Automatically saves best model: `best_model.pth`
- Generates training curves, confusion matrix, and other visualizations

**Output Files:**
- `best_model.pth`: Best model weights
- `scaler.pkl`: Data scaler
- `label_encoder.pkl`: Label encoder
- `confusion_matrix.png`: Confusion matrix
- `tsne_features.png`: t-SNE feature visualization
- `classification_radar.png`: Classification metrics radar chart

#### Step 8: Model Evaluation

```bash
python Pridict.py
```

**Evaluation Metrics:**
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

**Output:** Prediction results and evaluation reports saved in `saved_images/` directory

---

## 📊 Model Evaluation

### Performance Metrics

Performance on test set:

| Metric | Value |
|--------|-------|
| Accuracy | 99.86% |
| Precision | 99.84% |
| Recall | 99.83% |
| F1-Score | 99.83% |

### Comparative Experiments

| Method | Accuracy | Inference Speed (FPS) | Parameters |
|--------|----------|----------------------|------------|
| Traditional CNN | 87.26% | 25 | 2.1M |
| ResNet-50 | 92.34% | 18 | 25.6M |
| **Our Method** | **99.86%** | **32** | **0.8M** |

### Visualization Results

**1. Confusion Matrix**
- Shows prediction results for each class
- Identifies easily confused gesture pairs

**2. t-SNE Feature Distribution**
- Visualizes clustering in feature space
- Validates model's feature learning capability

**3. Attention Weight Heatmap**
- Shows keypoints the model focuses on
- Explains model decision-making basis

**4. Classification Metrics Radar Chart**
- Compares precision, recall, F1-score for each class
- Identifies directions for model optimization

---

## 🔌 Embedded Deployment

### Jetson Xavier NX Deployment

#### 1. Environment Configuration

```bash
# Install JetPack SDK (includes CUDA, cuDNN, TensorRT)
sudo apt-get update
sudo apt-get install nvidia-jetpack

# Verify installation
nvcc --version
```

#### 2. Model Conversion (PyTorch → ONNX → TensorRT)

**Convert to ONNX:**
```python
import torch

model = StaticGestureAttnModel(input_dim=3, num_classes=10)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

dummy_input = torch.randn(1, 42, 3)
torch.onnx.export(
    model,
    dummy_input,
    'model.onnx',
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}}
)
```

**Convert to TensorRT:**
```bash
trtexec --onnx=model.onnx \
        --saveEngine=model.trt \
        --fp16 \
        --workspace=2048
```

#### 3. Inference Optimization

**Key Optimization Techniques:**
- FP16 half-precision inference (2× speed boost)
- Batch inference
- Asynchronous input/output processing

**Performance Comparison:**

| Platform | Inference Speed | Model Size |
|----------|----------------|------------|
| PyTorch (CPU) | 8 FPS | 3.2 MB |
| PyTorch (GPU) | 45 FPS | 3.2 MB |
| TensorRT (Jetson) | 32 FPS | 1.0 MB |

---

## 📈 Experimental Results

### Accuracy Comparison

**Complex Background Testing:**

| Background Type | Our Method | Traditional CNN | ResNet-50 |
|----------------|------------|----------------|-----------|
| Solid Background | 99.92% | 92.15% | 95.67% |
| Indoor Scene | 99.86% | 87.26% | 91.43% |
| Outdoor Scene | 99.78% | 83.54% | 89.21% |
| Low Light | 99.65% | 79.32% | 85.76% |

### Ablation Study

| Model Configuration | Accuracy | Improvement |
|--------------------|----------|-------------|
| Baseline (no attention) | 94.23% | - |
| + Spatial Attention | 97.65% | +3.42% |
| + SMOTE | 98.87% | +4.64% |
| + Data Augmentation | **99.86%** | **+5.63%** |

### Real-time Performance Testing

**Inference Performance on Different Platforms:**

| Platform | Average Latency | FPS | Power |
|----------|----------------|-----|-------|
| RTX 3080 | 12ms | 83 | 320W |
| Jetson Xavier NX | 31ms | 32 | 15W |
| Jetson Nano | 78ms | 13 | 10W |

---
## 📚 Citation

If this project helps your research, please cite:
```
@dataset{guo_zhipeng_2025_17708961,
  author       = {Zhipeng Guo and Xiaobo Su and Caixia Zhou},
  title        = {Raw Video Dataset for Static Sign Language Recognition (9 Gesture Categories)},
  month        = nov,
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17708961},
  url          = {[https://doi.org/10.5281/zenodo.17708961](https://doi.org/10.5281/zenodo.17708961)}
}
```
---

## 📝 License

This project is licensed under the [MIT License](LICENSE).

---

## 🤝 Contributing

Issues and Pull Requests are welcome!

**Contribution Guidelines:**
1. Fork this repository
2. Create feature branch: `git checkout -b feature/AmazingFeature`
3. Commit changes: `git commit -m 'Add some AmazingFeature'`
4. Push to branch: `git push origin feature/AmazingFeature`
5. Submit Pull Request

---

## 📧 Contact

- Project Homepage: [https://github.com/Scatteredpeople/HandFind](https://github.com/Scatteredpeople/HandFind)
- Issue: [https://github.com/Scatteredpeople/HandFind/issues](https://github.com/Scatteredpeople/HandFind/issues)
- Email: guo.zp@outlook.com

---

## 🙏 Acknowledgments

- [MediaPipe](https://google.github.io/mediapipe/) - Hand keypoint detection
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [NVIDIA Jetson](https://developer.nvidia.com/embedded-computing) - Embedded platform support

