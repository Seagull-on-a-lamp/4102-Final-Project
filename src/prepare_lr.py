from pathlib import Path
from PIL import Image

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent

HR_DIR = PROJECT_ROOT / "datasets" / "test" / "DIV2K_valid_HR"
LR_DIR = PROJECT_ROOT / "datasets" / "test" / "DIV2K_valid_LR_bicubic_X2"

LR_DIR.mkdir(parents=True, exist_ok=True)

# Generate 2x low-resolution images
image_paths = sorted(HR_DIR.glob("*.png"))

for image_path in image_paths:
    with Image.open(image_path) as image:
        image = image.convert("RGB")

        lr_image = image.resize(
            (image.width // 2, image.height // 2),
            Image.Resampling.BICUBIC
        )

        lr_image.save(LR_DIR / image_path.name)

print(f"Generated {len(image_paths)} low-resolution images.")
print(f"Saved to: {LR_DIR}")