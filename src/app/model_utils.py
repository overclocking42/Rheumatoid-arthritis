import os
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[2]


class ANNModel(torch.nn.Module):
    """ANN checkpoint architecture for numeric inference."""

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(6, 32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, 3)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_numeric_model_artifacts():
    models_dir = ROOT / "models"
    model_path = models_dir / "ann_model.pth"
    scaler_path = models_dir / "ann_scaler.pkl"

    if not model_path.exists():
        return None, None

    checkpoint = torch.load(model_path, map_location="cpu")
    model = ANNModel()

    new_state_dict = {}
    for key, value in checkpoint.items():
        if key.startswith("0."):
            new_key = f"fc1.{key[2:]}"
        elif key.startswith("3."):
            new_key = f"fc2.{key[2:]}"
        elif key.startswith("6."):
            new_key = f"fc3.{key[2:]}"
        else:
            new_key = key
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    model.eval()

    scaler = None
    if scaler_path.exists():
        try:
            scaler = joblib.load(scaler_path)
        except Exception:
            import pickle

            with scaler_path.open("rb") as handle:
                scaler = pickle.load(handle)

    return model, scaler


def _build_swin_binary_classifier():
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    from transformers import SwinConfig, SwinModel

    config = SwinConfig(
        image_size=224,
        patch_size=4,
        num_channels=3,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        drop_path_rate=0.1,
        layer_norm_eps=1e-5,
        num_labels=1,
    )

    backbone = SwinModel(config)

    class SwinBinaryClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.swin = backbone
            self.classifier = torch.nn.Sequential(
                torch.nn.LayerNorm(config.hidden_size),
                torch.nn.Linear(config.hidden_size, 1),
            )

        def forward(self, x):
            outputs = self.swin(x)
            pooled = outputs.pooler_output
            return self.classifier(pooled)

    return SwinBinaryClassifier()


def load_image_model_artifact():
    ckpt = ROOT / "models" / "swin_fold4_best.pth"
    if not ckpt.exists():
        return None

    model = _build_swin_binary_classifier()
    checkpoint = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    return model


def preprocess_xray_image(img: Image.Image):
    tf = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Lambda(lambda im: im.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return tf(img)


def predict_xray_image(model, img_tensor):
    with torch.no_grad():
        logit = model(img_tensor.unsqueeze(0))
        prob_erosive = torch.sigmoid(logit[0, 0]).item()

    label = "Erosive" if prob_erosive >= 0.5 else "Non-Erosive"
    confidence = prob_erosive if label == "Erosive" else (1 - prob_erosive)
    return label, confidence


def normalize_numeric_features(raw_features, scaler):
    if scaler is not None:
        return scaler.transform([raw_features])[0].astype(np.float32)

    age, gender, rf, anti_ccp, crp, esr = raw_features
    return np.array(
        [
            age / 100.0,
            gender,
            rf / 500.0,
            anti_ccp / 500.0,
            crp / 100.0,
            esr / 100.0,
        ],
        dtype=np.float32,
    )
