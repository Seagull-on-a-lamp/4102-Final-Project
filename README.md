# Image Super Resolution
## Overview
This project compares a pretrained EDSR model with nearest-neighbour, bilinear, and bicubic interpolation for image super-resolution.
The model reconstructs higher-resolution images from low-resolution inputs using a scale factor of 2

It calculates RGB PSNR for each method on 100 DIV2K validation images and saves the results in CSV files.

## Project Structure
```
super-resolution/
├── datasets/
│   └── test/
│       ├── DIV2K_valid_HR/
│       └── DIV2K_valid_LR_bicubic_X2/
├── results/
│   ├── interpolation_psnr.csv
│   ├── edsr_psnr.csv
│   └── edsr/
│       └── reconstructed/
├── src/
│   ├── prepare_lr.py
│   ├── evaluate_interpolation.py
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
4. Activate the virtual environment (Windows PowerShell):
   ```
   .\.venv\Scripts\Activate.ps1
   ```
5. Install [PyTorch](https://pytorch.org/get-started/locally/) using the official installation guide, selecting a version compatible with your system and CUDA configuration.
6. Install the remaining Python dependencies:
    ```
    pip install numpy pillow opencv-python scikit-image tqdm matplotlib
    ```
7. Clone the EDSR implementation into the third_party directory:
    ```
    git clone https://github.com/sanghyun-son/EDSR-PyTorch.git third_party/EDSR-PyTorch
    ```
8. Download the pretrained weights (create the `weights` folder first):
    ```
    mkdir weights
    ```
    ```
    Invoke-WebRequest `
    -Uri "https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt" `
    -OutFile "weights/edsr_baseline_x2.pt"
    ```
9. Download the [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) validation [high-resolution dataset](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip) and extract its 100 images into datasets/test/DIV2K_valid_HR/.
10. Generate the corresponding low-resolution images by running:
    ```
    python .\src\prepare_lr.py
    ```

11. Run the interpolation methods from the project folder:
    ```
    python src/evaluate_interpolation.py
    ```
12. Run EDSR from inside `third_party/EDSR-PyTorch/src`:
    ```
    cd third_party/EDSR-PyTorch/src
    python main.py --model EDSR --data_test Demo --scale 2 --dir_demo ../../../datasets/test/DIV2K_valid_LR_bicubic_X2 --pre_train ../../../weights/edsr_baseline_x2.pt --test_only --save_results
    ```
    EDSR saves its outputs in its own experiment results folder. Copy the reconstructed images into `results/edsr/reconstructed/`, named like `0801_x2_SR.png`, before evaluating them.
13. Return to the project folder and calculate EDSR PSNR:
    ```
    cd ../../..
    python src/evaluate_edsr.py
    ```

## Average RGB PSNR (dB)

| Method | PSNR |
|---|---:|
| Nearest-neighbour | 29.2509 |
| Bilinear | 29.9340 |
| Bicubic | 31.2704 |
| EDSR | 34.6308 |
