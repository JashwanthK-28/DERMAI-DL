import io
import os
from typing import Dict, Any, Tuple
import cv2
import numpy as np
from PIL import Image, ImageOps
from fastapi import HTTPException, status
from backend.app.services.storage_service import StorageService


class PreprocessingService:
    """
    DermAI Image Preprocessing Engine (Section 3).
    Strict 8-step pipeline ensuring image quality, orientation, standardization, and normalization.
    """

    @staticmethod
    def process_image(raw_bytes: bytes, filename: str) -> Dict[str, Any]:
        # Step 1: Validate Non-Empty
        if not raw_bytes or len(raw_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file uploaded. Zero-byte payloads are rejected."
            )

        # Step 2: Validate & Decode Bytes via Pillow & OpenCV
        try:
            pil_img = Image.open(io.BytesIO(raw_bytes))
            pil_img.verify()  # Verifies file integrity
            # Reopen after verify because verify destroys buffer position
            pil_img = Image.open(io.BytesIO(raw_bytes))
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Corrupt or non-image file rejected: {str(e)}"
            )

        # Enforce minimum resolution (100x100)
        orig_width, orig_height = pil_img.size
        if orig_width < 100 or orig_height < 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Resolution ({orig_width}x{orig_height}) is below minimum required 100x100 pixels."
            )

        # Step 1 (Preserve Original): Save raw untouched bytes
        orig_storage_key = StorageService.generate_storage_key(filename, prefix="orig_")
        orig_path = StorageService.get_original_path(orig_storage_key)
        StorageService.save_bytes(raw_bytes, orig_path)

        # Step 3: EXIF Correction
        try:
            pil_img = ImageOps.exif_transpose(pil_img)
        except Exception:
            pass  # If no EXIF data present, continue with original

        # Convert PIL Image to OpenCV BGR numpy array
        pil_rgb = pil_img.convert("RGB")
        rgb_np = np.array(pil_rgb)
        bgr_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)

        # Step 4: Quality Assessment
        gray = cv2.cvtColor(bgr_np, cv2.COLOR_BGR2GRAY)
        
        # Blur Score: OpenCV Laplacian variance
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        
        # Brightness Score: Mean intensity
        brightness_score = float(np.mean(gray))
        
        # Contrast Score: Standard deviation of pixel intensities
        contrast_score = float(np.std(gray))

        quality_reasons = []
        is_warning = False
        is_reject = False

        if blur_score < 100.0:
            is_warning = True
            quality_reasons.append(f"BLURRY: Laplacian variance score ({blur_score:.1f}) < 100.0 threshold.")

        if brightness_score < 50.0:
            is_warning = True
            quality_reasons.append(f"UNDEREXPOSED: Mean intensity ({brightness_score:.1f}) < 50.0 threshold.")
        elif brightness_score > 200.0:
            is_warning = True
            quality_reasons.append(f"OVEREXPOSED: Mean intensity ({brightness_score:.1f}) > 200.0 threshold.")

        if contrast_score < 20.0:
            is_warning = True
            quality_reasons.append(f"LOW_CONTRAST: Standard deviation ({contrast_score:.1f}) < 20.0 threshold.")

        # Severe quality failure flags reject
        if blur_score < 20.0 or brightness_score < 20.0 or brightness_score > 240.0:
            is_reject = True
            quality_status = "REJECT"
        elif is_warning:
            quality_status = "WARNING"
        else:
            quality_status = "PASS"
            quality_reasons.append("Image satisfies sharpness, brightness, and contrast clinical quality thresholds.")

        # Step 5: Color Standardization (Ensure 3-channel RGB)
        # rgb_np is already 3-channel uint8 [0, 255]

        # Step 6: Spatial Resizing to 224x224 using bicubic interpolation
        resized_rgb = cv2.resize(rgb_np, (224, 224), interpolation=cv2.INTER_CUBIC)

        # Step 7: Pixel Normalization to float32 range [0.0, 1.0]
        normalized_tensor = resized_rgb.astype(np.float32) / 255.0

        # Step 8: Storage of processed 224x224 image
        processed_storage_key = StorageService.generate_storage_key(filename, prefix="proc_")
        processed_path = StorageService.get_preprocessed_path(processed_storage_key)
        
        # Save as standard PNG
        processed_bgr = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(processed_path, processed_bgr)

        return {
            "original_storage_key": orig_storage_key,
            "original_file_path": orig_path,
            "original_width": orig_width,
            "original_height": orig_height,
            "original_dim": f"{orig_width}x{orig_height}",
            "processed_storage_key": processed_storage_key,
            "processed_file_path": processed_path,
            "processed_dim": "224x224",
            "blur_score": round(blur_score, 2),
            "brightness_score": round(brightness_score, 2),
            "contrast_score": round(contrast_score, 2),
            "quality_status": quality_status,
            "quality_reasons": quality_reasons,
            "normalized_tensor": normalized_tensor,  # shape (224, 224, 3), float32 [0.0, 1.0]
        }
