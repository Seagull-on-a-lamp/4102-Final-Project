import csv
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio

ROOT = Path(__file__).resolve().parent.parent
HR_DIR = ROOT / 'datasets' / 'test' / 'DIV2K_valid_HR'
LR_DIR = ROOT / 'datasets' / 'test' / 'DIV2K_valid_LR_bicubic_X2'
RESULTS_DIR = ROOT / 'results'
PREVIEW_DIR = RESULTS_DIR / 'interpolation' / 'examples'
CSV_PATH = RESULTS_DIR / 'interpolation_psnr.csv'
METHODS = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
}


def main():
    
    images = sorted(HR_DIR.glob('*.png'))
    
    if not images:
        raise FileNotFoundError(f'No HR PNG images found in {HR_DIR}. Download the DIV2K validation HR images first.')
    
    if not LR_DIR.is_dir():
        raise FileNotFoundError(f'LR directory not found: {LR_DIR}. Run python src/prepare_lr.py first.')

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    
    for i, hr_path in enumerate(images):
        lr_path = LR_DIR / hr_path.name
        
        if not lr_path.exists():
            raise FileNotFoundError(f'Missing LR image {lr_path}. Run python src/prepare_lr.py.')
        hr_bgr = cv2.imread(str(hr_path), cv2.IMREAD_COLOR)
        lr_bgr = cv2.imread(str(lr_path), cv2.IMREAD_COLOR)
        
        if hr_bgr is None or lr_bgr is None:
            raise ValueError(f'Failed to load {hr_path.name}')
        
        if (lr_bgr.shape[1] * 2, lr_bgr.shape[0] * 2) != (hr_bgr.shape[1], hr_bgr.shape[0]):
            raise ValueError(f'Expected 2x LR/HR dimensions for {hr_path.name}; got LR {lr_bgr.shape}, HR {hr_bgr.shape}')
        hr_rgb = cv2.cvtColor(hr_bgr, cv2.COLOR_BGR2RGB)
        
        for method, interpolation in METHODS.items():
            upscale_bgr = cv2.resize(lr_bgr, (hr_bgr.shape[1], hr_bgr.shape[0]), interpolation=interpolation)
            upscale_rgb = cv2.cvtColor(upscale_bgr, cv2.COLOR_BGR2RGB)
            score = peak_signal_noise_ratio(hr_rgb, upscale_rgb, data_range=255)
            rows.append({'image': hr_path.name, 'method': method, 'psnr_db': float(score)})
            
            if i < 3:
                cv2.imwrite(str(PREVIEW_DIR / f'{hr_path.stem}_{method}.png'), upscale_bgr)
        print(f'Processed {i + 1}/{len(images)}: {hr_path.name}')

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    with CSV_PATH.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['image', 'method', 'psnr_db'])
        writer.writeheader()
        writer.writerows(rows)

    print('\nAverage RGB PSNR over all images:')
    
    for method in METHODS:
        scores = [r['psnr_db'] for r in rows if r['method'] == method]
        print(f'  {method:8s} {np.mean(scores):.4f} dB')
    print(f'\nCSV saved: {CSV_PATH}')
    print(f'Example images saved: {PREVIEW_DIR}')


if __name__ == '__main__':
    main()
