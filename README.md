# Image Super Resolution
## Overview
This project implements Image super resolution using a pretrained EDSR.
The model reconstructs higher-resolution images from low-resolution inputs using a scale factor of 2

It also calculates the PSNR for each image and saves it in a .csv file.

## Project Structure
```
super-resolution/
├── datasets/
│   └── test/
│       ├── DIV2K_valid_HR/
│       └── DIV2K_valid_LR_bicubic_X2/
├── results/
│   ├── edsr_psnr.csv
│   └── edsr/
│       └── reconstructed/
├── src/
│   ├── prepare_lr.py
│   └── evaluate_edsr.py
├── third_party/
│   └── EDSR-PyTorch/
├── weights/
│   └── edsr_baseline_x2.pt
├── .gitignore
└── README.md
```

## Setup 
1. Clone the repo
   ```
   git clone https://github.com/Seagull-on-a-lamp/4102-Final-Project.git
   ```
2. Navigate into the project directory:
   ```
   cd 4102-Final-Project
   ```
4. Create a python virtual environment:
   ```
   python
   ```
5. Activate the virtual environment:
   ```
   .\.venv\Scripts\Activate.ps1
   ```
6. Install [PyTorch](https://pytorch.org/get-started/locally/) using the official installation guide, selecting a version compatible with your system and CUDA configuration.
7. Install the remaining Python dependencies:
    ```
    pip install numpy pillow opencv-python scikit-image tqdm matplotlib
    ```
8. Clone the EDSR implementation into the third_party directory:
    ```
    git clone https://github.com/sanghyun-son/EDSR-PyTorch.git third_party/EDSR-PyTorch
    ```
9. Download the pretrained weights:
    ```
    Invoke-WebRequest `
    -Uri "https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt" `
    -OutFile "weights/edsr_baseline_x2.pt"
    ```
11. Download the [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) validation [high-resolution dataset](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip) and extract its 100 images into datasets/test/DIV2K_valid_HR/.
12. Generate the corresponding low-resolution images by running:
    ```
    python .\src\prepare_lr.py
    ```
