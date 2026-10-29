"""Import trained model from Colab/Google Drive.

Usage:
    python scripts/import_colab_checkpoint.py --checkpoint-path /path/to/tft_latest.pt

This copies the checkpoint from your Colab training into the project
and sets it up for the API to load.
"""

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Import TFT checkpoint from Colab")
    parser.add_argument(
        "--checkpoint-path", required=True, help="Path to trained .pt or .ckpt file"
    )
    parser.add_argument(
        "--reference-path", default=None, help="Optional path to tft_reference.parquet from Colab"
    )
    args = parser.parse_args()

    src_ckpt = Path(args.checkpoint_path)
    if not src_ckpt.exists():
        print(f"ERROR: Checkpoint not found: {src_ckpt}")
        return 1

    # Copy checkpoint
    dst_dir = Path("checkpoints/tft")
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_ckpt = dst_dir / "tft_latest.pt"
    shutil.copy2(src_ckpt, dst_ckpt)
    print(f"Copied checkpoint: {src_ckpt} -> {dst_ckpt}")

    # Optionally copy reference parquet
    if args.reference_path:
        src_ref = Path(args.reference_path)
        if src_ref.exists():
            dst_ref = Path("data/processed/tft_reference.parquet")
            dst_ref.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_ref, dst_ref)
            print(f"Copied reference: {src_ref} -> {dst_ref}")

    print("\nDone! Model is ready to use.")
    print("Restart the API or run: python scripts/smoke_test.py")
    return 0


if __name__ == "__main__":
    exit(main())
