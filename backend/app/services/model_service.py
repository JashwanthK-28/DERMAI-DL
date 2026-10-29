import os
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
from backend.app.config import settings
from ml.models import build_resnet50, build_efficientnet_b0, DIAGNOSTIC_CLASSES, CLASS_DESCRIPTIONS, RISK_MAPPING


class ModelService:
    _loaded_models: Dict[str, nn.Module] = {}
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ImageNet normalization isolated within model inference pipeline
    _inference_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    @classmethod
    def get_model(cls, model_name: str = "resnet50_baseline") -> nn.Module:
        if model_name in cls._loaded_models:
            return cls._loaded_models[model_name]

        device = cls._device

        if "efficientnet" in model_name.lower():
            model = build_efficientnet_b0(pretrained=False, num_classes=len(DIAGNOSTIC_CLASSES)).to(device)
            ckpt_path = os.path.join(settings.MODEL_DIR, "efficientnet_b0.pth")
            arch_name = "EfficientNet-B0"
        else:
            model = build_resnet50(pretrained=False, num_classes=len(DIAGNOSTIC_CLASSES)).to(device)
            ckpt_path = os.path.join(settings.MODEL_DIR, "resnet50_baseline.pth")
            arch_name = "ResNet50"

        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=device)
            state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
            model.load_state_dict(state_dict)
            print(f"[+] Loaded {arch_name} checkpoint from {ckpt_path}")
        else:
            print(f"[!] Warning: Checkpoint {ckpt_path} not found. Running with initialized {arch_name} weights.")

        model.eval()
        cls._loaded_models[model_name] = model
        return model

    @classmethod
    def predict(cls, image_input, model_name: str = "resnet50_baseline") -> Dict[str, Any]:
        """
        Run inference on image input (either file path, PIL Image, or 224x224 numpy array).
        Returns predicted class, probabilities summing to 1.0, and risk category.
        """
        model = cls.get_model(model_name)
        device = cls._device

        # Handle different input types
        if isinstance(image_input, str):
            # File path
            pil_img = Image.open(image_input).convert("RGB")
            if pil_img.size != (224, 224):
                pil_img = pil_img.resize((224, 224), Image.Resampling.BICUBIC)
            tensor = cls._inference_transform(pil_img).unsqueeze(0).to(device)
        elif isinstance(image_input, np.ndarray):
            # If shape is (224, 224, 3) and values in [0.0, 1.0]
            if image_input.dtype == np.float32 and image_input.max() <= 1.0:
                uint8_img = (image_input * 255.0).astype(np.uint8)
            else:
                uint8_img = image_input.astype(np.uint8)
            pil_img = Image.fromarray(uint8_img)
            tensor = cls._inference_transform(pil_img).unsqueeze(0).to(device)
        elif isinstance(image_input, torch.Tensor):
            if image_input.dim() == 3:
                tensor = image_input.unsqueeze(0).to(device)
            else:
                tensor = image_input.to(device)
        else:
            raise ValueError(f"Unsupported image_input type: {type(image_input)}")

        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()

        pred_idx = int(np.argmax(probs))
        predicted_class = DIAGNOSTIC_CLASSES[pred_idx]
        confidence_score = float(probs[pred_idx])

        class_probabilities = {
            cls_name: round(float(prob), 4)
            for cls_name, prob in zip(DIAGNOSTIC_CLASSES, probs)
        }

        # Risk category mapping
        risk_category = RISK_MAPPING.get(predicted_class, "MODERATE")

        return {
            "model_name": model_name,
            "model_version": "1.0.0",
            "predicted_class": predicted_class,
            "predicted_description": CLASS_DESCRIPTIONS.get(predicted_class, predicted_class),
            "confidence_score": round(confidence_score, 4),
            "class_probabilities": class_probabilities,
            "risk_category": risk_category,
        }
