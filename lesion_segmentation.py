"""
DermAI: Skin Disease Detector from Mobile Images Using Deep Learning
Task 3: Lesion Detection & Segmentation Investigation

1. Dataset Annotation Verification:
   Documents that PAD-UFES-20 contains clinical metadata and image files, but NO
   pixel-level ground-truth segmentation masks or bounding boxes exist.
2. Unsupervised / Zero-Shot Segmentation Technical Prototype:
   - Approach A: Otsu Thresholding + Morphological Filtering in LAB Color Space
   - Approach B: Saliency Map / Central-Prior GrabCut Segmentation
3. Visual Verification:
   Generates side-by-side artifact:
   [ Original Smartphone Image ] -> [ Generated Binary Mask ] -> [ Cropped Lesion ROI ]
"""

import os
import glob
from typing import Tuple, Dict, Any
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


def verify_dataset_annotations(metadata_path: str = "metadata_with_folds.csv"):
    """
    Examines PAD-UFES-20 annotation structure and verifies absence of ground-truth
    bounding boxes or pixel segmentation masks.
    """
    import pandas as pd
    df = pd.read_csv(metadata_path)
    columns = list(df.columns)

    mask_or_bbox_cols = [c for c in columns if any(k in c.lower() for k in ["mask", "bbox", "box", "segmentation", "poly", "coord"])]
    print("=" * 80)
    print("TASK 3 INVESTIGATION: PAD-UFES-20 ANNOTATION AUDIT")
    print("=" * 80)
    print(f"Total Dataset Images: {len(df)}")
    print(f"Available Metadata Columns ({len(columns)}): {columns}")
    print(f"Ground-Truth Mask or Bounding Box Columns Detected: {mask_or_bbox_cols}")
    print(
        "CONCLUSION: As documented by Pacheco et al. (2020), PAD-UFES-20 is a clinical mobile dataset "
        "designed for classification and contains NO pixel-level ground truth segmentation masks or "
        "bounding box coordinates. Consequently, lesion isolation requires unsupervised, morphological, "
        "or zero-shot region-of-interest (ROI) extraction prototypes."
    )
    print("=" * 80)


def approach_a_otsu_lab(image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Approach A: Otsu Thresholding + Morphological Filtering in LAB Color Space
    1. Convert image to CIELAB color space.
    2. Extract 'a*' channel (representing green-red chrominance, ideal for erythematous & pigmented lesions).
    3. Apply Gaussian blur to suppress fine skin textures / hair noise.
    4. Apply Otsu's thresholding to compute optimal global threshold.
    5. Perform morphological closing (dilation then erosion) with elliptical structuring element.
    6. Extract the largest connected contour to produce the final clean binary lesion mask.
    7. Mask and crop to lesion Region of Interest (ROI).
    """
    # 1. Convert to LAB color space
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # 2. Gaussian blur on 'a*' channel to reduce skin texture noise
    blurred = cv2.GaussianBlur(a_channel, (5, 5), 0)

    # 3. Otsu automatic thresholding
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert if majority of pixels are white (background should be black)
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)

    # 4. Morphological closing (remove holes) followed by opening (remove small noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    # 5. Find contours and select largest area (the primary lesion)
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(opened)

    bbox = None
    if contours:
        largest_cnt = max(contours, key=cv2.contourArea)
        cv2.drawContours(clean_mask, [largest_cnt], -1, 255, thickness=cv2.FILLED)
        x, y, w, h = cv2.boundingRect(largest_cnt)
        # Pad bounding box slightly
        pad = 10
        h_img, w_img = image_bgr.shape[:2]
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w_img, x + w + pad)
        y2 = min(h_img, y + h + pad)
        bbox = (x1, y1, x2 - x1, y2 - y1)
    else:
        # Fallback to center box if no contour found
        h_img, w_img = image_bgr.shape[:2]
        clean_mask[h_img//4:3*h_img//4, w_img//4:3*w_img//4] = 255
        bbox = (w_img//4, h_img//4, w_img//2, h_img//2)

    # 6. Apply mask and crop ROI
    masked_img = cv2.bitwise_and(image_bgr, image_bgr, mask=clean_mask)
    x, y, w, h = bbox
    cropped_roi = image_bgr[y:y+h, x:x+w]

    stats = {
        "method": "LAB Otsu + Morphology",
        "lesion_area_pixels": int(np.sum(clean_mask == 255)),
        "mask_coverage_pct": round(float(np.mean(clean_mask == 255) * 100), 2),
        "bounding_box": bbox,
    }
    return clean_mask, cropped_roi, stats


def approach_b_grabcut(image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Approach B: Central-Prior GrabCut Foreground Extraction
    1. Initialize a central rectangular bounding box assuming mobile photo centers on the lesion.
    2. Initialize background and foreground Gaussian Mixture Models (GMMs).
    3. Run cv2.grabCut iterations to partition lesion foreground from healthy skin background.
    4. Generate binary mask from GC_FGD and GC_PR_FGD pixels.
    5. Crop lesion ROI.
    """
    h, w = image_bgr.shape[:2]
    # Margin of 12% around edges as definite background prior
    margin_x = int(w * 0.12)
    margin_y = int(h * 0.12)
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(image_bgr, mask, rect, bgd_model, fgd_model, iterCount=5, mode=cv2.GC_INIT_WITH_RECT)
        # Probable and definite foreground are 1 and 3
        binary_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype("uint8")
    except Exception as e:
        print(f"[!] GrabCut fallback due to: {e}")
        binary_mask = np.zeros((h, w), np.uint8)
        binary_mask[margin_y:h-margin_y, margin_x:w-margin_x] = 255

    # Refine mask with morphological opening to remove stray pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    # Find bounding box
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_cnt = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(largest_cnt)
        cropped_roi = image_bgr[y:y+bh, x:x+bw]
        bbox = (x, y, bw, bh)
    else:
        cropped_roi = image_bgr[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
        bbox = rect

    stats = {
        "method": "Central-Prior GrabCut",
        "lesion_area_pixels": int(np.sum(binary_mask == 255)),
        "mask_coverage_pct": round(float(np.mean(binary_mask == 255) * 100), 2),
        "bounding_box": bbox,
    }
    return binary_mask, cropped_roi, stats


def generate_visual_comparison_artifacts(image_paths: list, output_dir: str = "outputs/segmentation"):
    """
    Renders visual comparisons:
    [ Original Smartphone Image ] -> [ Generated Binary Mask ] -> [ Cropped Lesion ROI ]
    for both Approach A and Approach B across representative clinical samples.
    """
    os.makedirs(output_dir, exist_ok=True)

    for idx, path in enumerate(image_paths):
        img_name = os.path.basename(path)
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            continue

        # Resize to standard 300x300 for clean visualization
        img_bgr = cv2.resize(img_bgr, (300, 300), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Execute Approach A
        mask_a, roi_a, stats_a = approach_a_otsu_lab(img_bgr)
        roi_a_rgb = cv2.cvtColor(roi_a, cv2.COLOR_BGR2RGB) if roi_a.size > 0 else img_rgb

        # Execute Approach B
        mask_b, roi_b, stats_b = approach_b_grabcut(img_bgr)
        roi_b_rgb = cv2.cvtColor(roi_b, cv2.COLOR_BGR2RGB) if roi_b.size > 0 else img_rgb

        # Plot 2-row comparison: Row 1 = Approach A, Row 2 = Approach B
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))

        # Row 1: Approach A (LAB Otsu + Morphology)
        axes[0, 0].imshow(img_rgb)
        axes[0, 0].set_title(f"Original Mobile Image\n({img_name})", fontsize=11, fontweight="bold")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(mask_a, cmap="gray")
        axes[0, 1].set_title(f"Approach A: Binary Mask\n(LAB Otsu + Morph, {stats_a['mask_coverage_pct']}%)", fontsize=11, fontweight="bold")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(roi_a_rgb)
        axes[0, 2].set_title(f"Approach A: Cropped Lesion ROI\n(Bounding Box: {stats_a['bounding_box'][2]}x{stats_a['bounding_box'][3]})", fontsize=11, fontweight="bold")
        axes[0, 2].axis("off")

        # Row 2: Approach B (Central-Prior GrabCut)
        axes[1, 0].imshow(img_rgb)
        axes[1, 0].set_title(f"Original Mobile Image\n({img_name})", fontsize=11, fontweight="bold")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(mask_b, cmap="gray")
        axes[1, 1].set_title(f"Approach B: Binary Mask\n(Central-Prior GrabCut, {stats_b['mask_coverage_pct']}%)", fontsize=11, fontweight="bold")
        axes[1, 1].axis("off")

        axes[1, 2].imshow(roi_b_rgb)
        axes[1, 2].set_title(f"Approach B: Cropped Lesion ROI\n(Bounding Box: {stats_b['bounding_box'][2]}x{stats_b['bounding_box'][3]})", fontsize=11, fontweight="bold")
        axes[1, 2].axis("off")

        plt.suptitle(
            "Task 3: DermAI Unsupervised Skin Lesion Detection & Segmentation Prototype\n"
            "[ Original Smartphone Image ] -> [ Generated Binary Mask ] -> [ Cropped Lesion ROI ]",
            fontsize=13, fontweight="bold", y=0.98
        )
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"segmentation_demo_{idx+1}_{os.path.splitext(img_name)[0]}.png")
        plt.savefig(save_path, dpi=180)
        plt.close()
        print(f"[+] Saved visual comparison artifact: {save_path}")


def main():
    metadata_path = "metadata_with_folds.csv" if os.path.exists("metadata_with_folds.csv") else "data/metadata_with_folds.csv"
    verify_dataset_annotations(metadata_path)

    # Select representative samples from test set (including BCC, MEL, NEV)
    import pandas as pd
    df = pd.read_csv(metadata_path)
    sample_images = []
    
    # Search for files on disk
    search_dirs = [
        "data/images/imgs_part_1", "data/images/imgs_part_2", "data/images/imgs_part_3",
        "/home/user/Documents/DermAI Project/images/imgs_part_1",
        "/home/user/Documents/DermAI Project/images/imgs_part_2",
        "/home/user/Documents/DermAI Project/images/imgs_part_3",
    ]

    for diag in ["BCC", "MEL", "NEV"]:
        diag_rows = df[df["diagnostic"] == diag]
        for _, row in diag_rows.iterrows():
            img_id = row["img_id"]
            for d in search_dirs:
                cand = os.path.join(d, img_id)
                if os.path.exists(cand):
                    sample_images.append(cand)
                    break
            if len(sample_images) >= 3:
                break
        if len(sample_images) >= 3:
            break

    print(f"\n[*] Processing {len(sample_images)} clinical samples for segmentation prototype visual verification...")
    generate_visual_comparison_artifacts(sample_images, "outputs/segmentation")
    print("\n[+] Task 3 Lesion Detection & Segmentation Prototype completed successfully!")


if __name__ == "__main__":
    main()
