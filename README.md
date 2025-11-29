# Sign Language Detection Using YOLOv5

This repository contains a Jupyter notebook which demonstrates sign language detection using the YOLOv5 object detection framework.

**Project**: Sign Language Detection

**Main file**: `Sign_language_Detection_Using_YOLO_v5.ipynb`

## Overview
- **Goal:** Detect hand signs (sign language gestures) in images/video using a YOLOv5-based model.
- **Approach:** The notebook shows preprocessing, dataset preparation, training (or using a pretrained YOLOv5 model), and inference/visualization steps.

## Prerequisites
- Python 3.8+ (3.10/3.11 recommended)
- Git
- Jupyter Notebook or VS Code with Jupyter support
- GPU recommended for training (CUDA-enabled NVIDIA GPU)

## Recommended Python packages
Install packages in a virtual environment. Example with `pip`:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Install common dependencies (adjust for your platform/CUDA version)
python -m pip install notebook jupyterlab matplotlib opencv-python pandas seaborn tqdm pillow scikit-learn
# Install PyTorch (follow instructions at https://pytorch.org/ for correct CUDA/cuDNN)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
# Clone YOLOv5 and install its requirements if using the repo
# git clone https://github.com/ultralytics/yolov5.git
# python -m pip install -r yolov5/requirements.txt
```

Note: Replace the PyTorch install command with the correct one for your OS and CUDA version. If you plan to use a CPU-only setup, install the CPU wheels from PyTorch instructions.

## Quick Start
1. Open the notebook `Sign_language_Detection_Using_YOLO_v5.ipynb` in Jupyter or VS Code.
2. Follow the notebook cells in order. The notebook should include sections for:
   - Dataset loading and annotation format
   - Model configuration (YOLOv5)
   - Training or fine-tuning steps
   - Inference and visualization
3. If you use YOLOv5 from the Ultralytics repo, ensure the notebook imports the `yolov5` code (either by cloning the repo or installing a package with the necessary modules).

## Dataset
- The notebook expects a dataset of hand sign images/frames and annotations in a YOLO-compatible format (text files with bounding boxes normalized to image dimensions) or COCO format depending on the notebook's implementation.
- If you don't have a dataset, consider using an existing sign-language dataset or prepare your own by labeling images with tools like LabelImg or Roboflow.

## Training & Inference
- Training: Configure hyperparameters and dataset paths in the notebook, then run the training cell which typically calls YOLOv5's `train.py` or equivalent training function.
- Inference: Use the trained `.pt` model to run detection on images or video, and visualize predictions.

## Files in this repo
- `Sign_language_Detection_Using_YOLO_v5.ipynb` — The main notebook demonstrating the pipeline.
- `LICENSE` — Project license file.

## Credits
- YOLOv5 — Ultralytics (https://github.com/ultralytics/yolov5)
- Author / Maintainer: murad-hussain-sarker

## License
See the `LICENSE` file in this repository for license terms.

## Contact
If you have questions or want help running the notebook, contact the repository owner.
