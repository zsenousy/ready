"""
Main script for resizing and augmenting image-mask pairs.
"""

from pathlib import Path
from PIL import Image
import argparse
import random

from ready.preprocessing.utils import find_pairs
from ready.preprocessing.geometric import apply_geometric
from ready.preprocessing.photometric import apply_photometric


def main():
    parser = argparse.ArgumentParser(description="Resize and augment image-mask pairs.")

    parser.add_argument("--image-dir", required=True, help="folder with input images")
    parser.add_argument("--mask-dir", required=True, help="folder with input masks")
    parser.add_argument("--output-dir", required=True, help="folder to save the new files")

    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=400)
    parser.add_argument("--limit", type=int, default=None, help="only process this many pairs")

    for name in ["hflip", "vflip", "rot90", "rot180", "rot270"]:
        parser.add_argument(f"--{name}", action="store_true")

    parser.add_argument("--brightness-up", action="store_true")
    parser.add_argument("--brightness-up-factor", type=float, default=1.5)

    parser.add_argument("--brightness-down", action="store_true")
    parser.add_argument("--brightness-down-factor", type=float, default=0.6)

    parser.add_argument("--contrast-up", action="store_true")
    parser.add_argument("--contrast-up-factor", type=float, default=1.5)

    parser.add_argument("--blur", action="store_true")
    parser.add_argument("--blur-factor", type=float, default=2.0)

    parser.add_argument("--grayscale", action="store_true")

    parser.add_argument("--autocontrast", action="store_true")
    parser.add_argument("--autocontrast-factor", type=float, default=0.0)

    parser.add_argument("--equalize", action="store_true")

    parser.add_argument("--gamma", action="store_true")
    parser.add_argument("--gamma-factor", type=float, default=1.0)

    parser.add_argument("--clahe", action="store_true")
    parser.add_argument("--clahe-clip-limit", type=float, default=1.5)
    parser.add_argument("--clahe-tile-grid-size", type=int, default=8)

    args = parser.parse_args()

    output_images = Path(args.output_dir) / "images"
    output_masks = Path(args.output_dir) / "masks"

    output_images.mkdir(parents=True, exist_ok=True)
    output_masks.mkdir(parents=True, exist_ok=True)

    pairs, skipped = find_pairs(args.image_dir, args.mask_dir)

    if args.limit is not None:
        pairs = random.sample(pairs, min(args.limit, len(pairs)))

    geometric_saved = 0
    photometric_saved = 0

    for image_path, mask_path in pairs:
        stem = image_path.stem
        image_ext = image_path.suffix
        mask_ext = mask_path.suffix

        try:
            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path)
            image = image.resize((args.width, args.height), Image.Resampling.BILINEAR)
            mask = mask.resize((args.width, args.height), Image.Resampling.NEAREST)
        except Exception as e:
            print(f"Skipping corrupted pair: {image_path.name} — {e}")
            skipped.append((image_path.name, f"corrupted file: {e}"))
            continue

        image.save(output_images / f"{stem}{image_ext}")
        mask.save(output_masks / f"{stem}{mask_ext}")

        for name, new_image, new_mask in apply_geometric(image, mask, args):
            new_image.save(output_images / f"{stem}_{name}{image_ext}")
            new_mask.save(output_masks / f"{stem}_{name}{mask_ext}")
            geometric_saved += 1

        for name, new_image in apply_photometric(image, args):
            new_image.save(output_images / f"{stem}_{name}{image_ext}")
            mask.save(output_masks / f"{stem}_{name}{mask_ext}")
            photometric_saved += 1

    total_saved = len(pairs) + geometric_saved + photometric_saved

    print("\nDone.")
    print("Saved to:", args.output_dir)
    print("Original pairs saved:", len(pairs))
    print("Geometric versions saved:", geometric_saved)
    print("Image-only versions saved:", photometric_saved)
    print("Total pairs saved:", total_saved)

    if skipped:
        print(f"\nSkipped {len(skipped)} files:")
        for filename, reason in skipped:
            print(f"- {filename}: {reason}")


if __name__ == "__main__":
    main()
