import csv
import cv2
import numpy as np
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent

HR_DIR = PROJECT_ROOT / "datasets" / "test" / "DIV2K_valid_HR"

SR_DIR = (
    PROJECT_ROOT
    / "results"
    / "edsr"
    / "reconstructed"
)

OUTPUT_CSV = PROJECT_ROOT / "results" / "edsr_psnr.csv"

results = []

# Evaluate every original validation image
image_paths = sorted(HR_DIR.glob("*.png"))

for hr_path in image_paths:
    image_name = hr_path.name
    sr_path = SR_DIR / f"{hr_path.stem}_x2_SR.png"

    # Check that the reconstructed image exists
    if not sr_path.exists():
        raise FileNotFoundError(f"Missing reconstruction: {sr_path}")

    # Load images
    hr = cv2.imread(str(hr_path))
    sr = cv2.imread(str(sr_path))

    if hr is None or sr is None:
        raise ValueError(f"Could not load images for {image_name}")

    # Convert BGR to RGB
    hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
    sr = cv2.cvtColor(sr, cv2.COLOR_BGR2RGB)

    # Check dimensions
    if hr.shape != sr.shape:
        raise ValueError(
            f"Dimension mismatch for {image_name}: "
            f"HR={hr.shape}, SR={sr.shape}"
        )

    # Calculate RGB PSNR
    psnr = peak_signal_noise_ratio(
        hr,
        sr,
        data_range=255
    )

    results.append({
        "image": image_name,
        "psnr_db": psnr
    })

# Calculate summary statistics
psnr_values = [result["psnr_db"] for result in results]

mean_psnr = np.mean(psnr_values)
min_psnr = np.min(psnr_values)
max_psnr = np.max(psnr_values)

# Save individual image results
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_CSV, "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["image", "psnr_db"]
    )
    writer.writeheader()
    writer.writerows(results)

# Display results
print(f"Images evaluated: {len(results)}")
print(f"Average RGB PSNR: {mean_psnr:.4f} dB")
print(f"Minimum RGB PSNR: {min_psnr:.4f} dB")
print(f"Maximum RGB PSNR: {max_psnr:.4f} dB")
print(f"\nResults saved to: {OUTPUT_CSV}")
