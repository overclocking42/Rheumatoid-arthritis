import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "app"))

from model_utils import (  # noqa: E402
    load_image_model_artifact,
    load_numeric_model_artifacts,
    normalize_numeric_features,
    predict_xray_image,
    preprocess_xray_image,
)


def test_numeric():
    model, scaler = load_numeric_model_artifacts()
    if model is None:
        raise RuntimeError("Numeric model not found at models/ann_model.pth")

    sample = [55.0, 1.0, 25.0, 15.0, 8.0, 30.0]
    features = normalize_numeric_features(sample, scaler)
    tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        pred = int(torch.argmax(logits, dim=1).item())
    print(f"numeric_ok class_index={pred}")


def test_image(image_path: Path):
    model = load_image_model_artifact()
    if model is None:
        raise RuntimeError("Image model not found at models/swin_fold4_best.pth")

    img = Image.open(image_path)
    tensor = preprocess_xray_image(img)
    label, confidence = predict_xray_image(model, tensor)
    print(f"image_ok label={label} confidence={confidence:.4f} file={image_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Smoke-test local inference on macOS.")
    parser.add_argument(
        "--image",
        default="data_holdout/image_judge_samples/patient1.jpeg",
        help="Path to a sample image for X-ray inference.",
    )
    args = parser.parse_args()

    test_numeric()
    test_image(ROOT / args.image)


if __name__ == "__main__":
    main()
