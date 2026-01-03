"""
==============================================================================
SMART CROP-SEGMENT PIPELINE
GroundingDINO → Crop Box → SAM2 Segment → Paste Back → LaMa Inpaint

Pipeline Flow:
1. DINO detects bounding box around object
2. Crop image to box region (small area)
3. SAM2 segments object in cropped region (accurate!)
4. Paste mask back to original position
5. LaMa inpaints the masked region

Benefits:
- SAM2 only processes small cropped region → Faster & More accurate
- Auto-select largest mask (main object)
- No box conversion issues
==============================================================================
"""

import sys
import os
import cv2
import numpy as np
from typing import Tuple, List
import warnings

warnings.filterwarnings('ignore')

# =================================================================
# CẤU HÌNH ĐƯỜNG DẪN TỰ ĐỘNG (RELATIVE PATH)
# =================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Thêm đường dẫn source code GroundingDINO
gd_path = os.path.join(BASE_DIR, "modules", "grounding", "GroundingDINO")
if gd_path not in sys.path:
    sys.path.append(gd_path)

# Thêm đường dẫn package sam2
sam2_pkg_path = os.path.join(BASE_DIR, "modules", "segmentation", "sam2")
if sam2_pkg_path not in sys.path:
    sys.path.append(sam2_pkg_path)

# Thêm modules root
modules_root = os.path.join(BASE_DIR, "modules")
if modules_root not in sys.path:
    sys.path.append(modules_root)

# Thêm segmentation directory
segmentation_dir = os.path.join(BASE_DIR, "modules", "segmentation")
if segmentation_dir not in sys.path:
    sys.path.append(segmentation_dir)

# =================================================================
# IMPORT MODULES
# =================================================================
try:
    from modules.grounding.groundingDINO import GroundingDINOStrategy
    from modules.segmentation.sam2_mask_strategy import Sam2MaskStrategy
    from modules.inpainting.deep_strategies import DeepInpaintingStrategy
except ImportError as e:
    print(f"❌ Lỗi Import: {e}")
    print(f"👉 Vui lòng kiểm tra xem thư mục này có tồn tại không: {gd_path}")
    sys.exit(1)


# =================================================================
# UTILITY FUNCTIONS
# =================================================================

def convert_box_to_corners(box: np.ndarray, image_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    """
    Convert DINO box (cx, cy, w, h) to corner coordinates (x1, y1, x2, y2)
    Auto-detect normalized vs absolute format
    
    Args:
        box: Box in format [cx, cy, w, h]
        image_shape: (height, width, channels)
    
    Returns:
        (x1, y1, x2, y2) in absolute pixel coordinates
    """
    h_img, w_img = image_shape[:2]
    cx, cy, w, h = box
    
    # Auto-detect format
    is_normalized = all(v <= 10.0 for v in [cx, cy, w, h])
    
    # Convert to absolute if needed
    if is_normalized:
        cx_abs = cx * w_img
        cy_abs = cy * h_img
        w_abs = w * w_img
        h_abs = h * h_img
    else:
        cx_abs, cy_abs = cx, cy
        w_abs, h_abs = w, h
    
    # Convert to corners
    x1 = int(max(0, cx_abs - w_abs / 2))
    y1 = int(max(0, cy_abs - h_abs / 2))
    x2 = int(min(w_img, cx_abs + w_abs / 2))
    y2 = int(min(h_img, cy_abs + h_abs / 2))
    
    return x1, y1, x2, y2


def crop_image_with_padding(image: np.ndarray, 
                            x1: int, y1: int, x2: int, y2: int,
                            padding: int = 20) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Crop image with padding around the box
    
    Args:
        image: Input image
        x1, y1, x2, y2: Box coordinates
        padding: Padding pixels around box
    
    Returns:
        cropped_image: Cropped region
        actual_coords: (x1_new, y1_new, x2_new, y2_new) coordinates used
    """
    h, w = image.shape[:2]
    
    # Add padding
    x1_padded = max(0, x1 - padding)
    y1_padded = max(0, y1 - padding)
    x2_padded = min(w, x2 + padding)
    y2_padded = min(h, y2 + padding)
    
    # Crop
    cropped = image[y1_padded:y2_padded, x1_padded:x2_padded]
    
    return cropped, (x1_padded, y1_padded, x2_padded, y2_padded)


def segment_cropped_region(cropped_rgb: np.ndarray, sam2_model) -> np.ndarray:
    """
    Segment object in cropped region using SAM2
    Automatically selects the largest mask (main object)
    
    Args:
        cropped_rgb: Cropped image in RGB format
        sam2_model: SAM2 model instance
    
    Returns:
        Binary mask for the cropped region
    """
    h_crop, w_crop = cropped_rgb.shape[:2]
    
    print(f"      Cropped region size: {w_crop}x{h_crop}")
    
    # Use SAM2 with automatic mask generation (no box prompt)
    # Or use the entire crop as a box
    crop_box = np.array([[0, 0, w_crop, h_crop]], dtype=np.float32)
    
    print(f"      Running SAM2 on cropped region...")
    raw_mask = sam2_model.process(cropped_rgb, boxes=crop_box)
    
    # Process mask
    if hasattr(raw_mask, 'cpu'):
        mask = raw_mask.cpu().detach().numpy()
    else:
        mask = np.array(raw_mask)
    
    print(f"      Raw mask shape: {mask.shape}")
    
    # Handle dimensions
    while mask.ndim > 2 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    
    # If multiple masks, select the largest one
    if mask.ndim == 3:
        print(f"      Found {mask.shape[0]} mask(s), selecting largest...")
        
        # Calculate area for each mask
        mask_areas = []
        for i in range(mask.shape[0]):
            single_mask = mask[i]
            area = np.sum(single_mask > 0)
            mask_areas.append(area)
            print(f"         Mask {i+1}: {area:,} pixels")
        
        # Select largest mask
        largest_idx = np.argmax(mask_areas)
        mask = mask[largest_idx]
        print(f"      ✅ Selected mask {largest_idx + 1} (largest)")
    
    # Resize if needed
    if mask.shape[:2] != (h_crop, w_crop):
        print(f"      Resizing mask: {mask.shape[0]}x{mask.shape[1]} → {h_crop}x{w_crop}")
        mask = cv2.resize(mask, (w_crop, h_crop), interpolation=cv2.INTER_LINEAR)
    
    # Binarize
    mask = (mask > 0.0).astype(np.uint8) * 255
    
    # Get statistics
    mask_pixels = np.sum(mask > 0)
    total_pixels = h_crop * w_crop
    coverage = (mask_pixels / total_pixels) * 100
    
    print(f"      ✅ Mask coverage: {coverage:.2f}% ({mask_pixels:,} pixels)")
    
    return mask


def paste_mask_to_original(original_shape: Tuple[int, int, int],
                           cropped_mask: np.ndarray,
                           crop_coords: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Paste cropped mask back to original image size
    
    Args:
        original_shape: (height, width, channels) of original image
        cropped_mask: Mask from cropped region
        crop_coords: (x1, y1, x2, y2) coordinates where crop was taken
    
    Returns:
        Full-size mask
    """
    h_orig, w_orig = original_shape[:2]
    x1, y1, x2, y2 = crop_coords
    
    # Create empty mask
    full_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
    
    # Paste cropped mask
    full_mask[y1:y2, x1:x2] = cropped_mask
    
    print(f"      ✅ Pasted mask back to original position [{x1}, {y1}, {x2}, {y2}]")
    
    return full_mask


def refine_mask(mask: np.ndarray, 
               close_kernel: int = 15,
               dilate_kernel: int = 20) -> np.ndarray:
    """
    Refine mask with morphological operations
    
    Args:
        mask: Binary mask
        close_kernel: Kernel size for closing
        dilate_kernel: Kernel size for dilation
    
    Returns:
        Refined mask
    """
    # Closing: Fill small holes
    kernel_close = np.ones((close_kernel, close_kernel), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    
    # Dilation: Expand slightly for better inpainting
    kernel_dilate = np.ones((dilate_kernel, dilate_kernel), np.uint8)
    mask = cv2.dilate(mask, kernel_dilate, iterations=1)
    
    # Smooth edges
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = (mask > 127).astype(np.uint8) * 255
    
    return mask


def visualize_pipeline_step(original: np.ndarray,
                           detection_boxes: List[Tuple[int, int, int, int]],
                           cropped_regions: List[np.ndarray],
                           masks: List[np.ndarray],
                           final_mask: np.ndarray) -> np.ndarray:
    """
    Create visualization of the pipeline steps
    
    Returns:
        Visualization image
    """
    h, w = original.shape[:2]
    
    # Panel 1: Original with detection boxes
    panel1 = original.copy()
    for i, (x1, y1, x2, y2) in enumerate(detection_boxes):
        cv2.rectangle(panel1, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(panel1, f"#{i+1}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Panel 2: Cropped regions (stack vertically if multiple)
    if cropped_regions:
        # Resize all crops to same width
        max_w = max(crop.shape[1] for crop in cropped_regions)
        resized_crops = []
        for crop in cropped_regions:
            h_crop = int(crop.shape[0] * max_w / crop.shape[1])
            resized = cv2.resize(crop, (max_w, h_crop))
            resized_crops.append(resized)
        
        panel2 = cv2.vconcat(resized_crops)
        # Resize to match original height
        panel2 = cv2.resize(panel2, (int(panel2.shape[1] * h / panel2.shape[0]), h))
    else:
        panel2 = np.zeros_like(original)
    
    # Panel 3: Final mask
    panel3 = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
    
    # Combine
    combined = cv2.hconcat([panel1, panel2, panel3])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    labels = [
        ("1. Detection", (20, 40), (0, 255, 0)),
        ("2. Cropped Regions", (w + 20, 40), (255, 255, 0)),
        ("3. Final Mask", (w * 2 + 20, 40), (255, 255, 255))
    ]
    
    for text, pos, color in labels:
        cv2.putText(combined, text, pos, font, 1.2, (0, 0, 0), 4)
        cv2.putText(combined, text, pos, font, 1.2, color, 2)
    
    return combined


# =================================================================
# MAIN PIPELINE
# =================================================================

def main():
    print("\n" + "="*70)
    print("🚀 SMART CROP-SEGMENT INPAINTING PIPELINE")
    print("   DINO → Crop → SAM2 → Paste → LaMa")
    print("="*70)
    
    # =================================================================
    # 0. CẤU HÌNH INPUT/OUTPUT
    # =================================================================
    img_path = os.path.join(BASE_DIR, "inputs", "test_image1.jpg")
    output_dir = os.path.join(BASE_DIR, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    # Weights Config
    dino_config = os.path.join(BASE_DIR, "weights", "GroundingDINO_SwinB_cfg.py")
    dino_weights = os.path.join(BASE_DIR, "weights", "groundingdino_swinb_cogcoor.pth")
    sam2_checkpoint = os.path.join(BASE_DIR, "weights", "sam2_hiera_base_plus.pt")
    sam2_config = "sam2_hiera_b+.yaml"
    lama_model_path = os.path.join(BASE_DIR, "weights", "big-lama.pt")
    
    # User input
    print(f"\n📁 Input: {os.path.basename(img_path)}")
    text_prompt = input("👉 Nhập vật thể muốn xóa: ").strip()
    
    if not text_prompt:
        print("❌ Không có prompt!")
        return
    
    # Check image
    if not os.path.exists(img_path):
        print(f"❌ Không tìm thấy file tại {img_path}")
        return
    
    # Load image
    original_image = cv2.imread(img_path)
    h_orig, w_orig = original_image.shape[:2]
    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    print(f"✅ Loaded: {w_orig}x{h_orig} pixels")
    
    # =================================================================
    # BƯỚC 1: DETECTION (GroundingDINO)
    # =================================================================
    print(f"\n{'='*70}")
    print("🎯 BƯỚC 1: DETECTION (GroundingDINO)")
    print("="*70)
    
    try:
        if not os.path.exists(dino_config) or not os.path.exists(dino_weights):
            print("❌ Thiếu file GroundingDINO trong thư mục weights/")
            return
        
        print("   📦 Loading GroundingDINO...")
        detector = GroundingDINOStrategy(
            config_path=dino_config,
            weights_path=dino_weights,
            device=None
        )
        
        print(f"   🔍 Đang tìm: '{text_prompt}'...")
        boxes, scores = detector.detect(original_image, text_prompt=text_prompt)
        
        if len(boxes) == 0:
            print("❌ Không tìm thấy vật thể!")
            return
        
        print(f"\n   ✅ Tìm thấy {len(boxes)} vật thể:")
        
        # Convert all boxes to corner format
        detection_boxes = []
        for i, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = convert_box_to_corners(box, original_image.shape)
            detection_boxes.append((x1, y1, x2, y2))
            print(f"      Box {i+1}: [{x1}, {y1}, {x2}, {y2}] | conf={score:.3f}")
        
    except Exception as e:
        print(f"❌ Lỗi Detection: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # =================================================================
    # BƯỚC 2: CROP & SEGMENT (SAM2)
    # =================================================================
    print(f"\n{'='*70}")
    print("✂️  BƯỚC 2: CROP & SEGMENT (SAM2)")
    print("="*70)
    
    try:
        print("   📦 Loading SAM2...")
        sam2 = Sam2MaskStrategy(
            checkpoint_path=sam2_checkpoint,
            config_path=sam2_config
        )
        
        cropped_regions = []
        cropped_masks = []
        crop_coords_list = []
        
        for i, (x1, y1, x2, y2) in enumerate(detection_boxes):
            print(f"\n   🔸 Processing box {i+1}/{len(detection_boxes)}:")
            
            # Crop with padding
            print(f"      📐 Cropping region [{x1}, {y1}, {x2}, {y2}] + padding...")
            cropped_bgr, crop_coords = crop_image_with_padding(
                original_image, x1, y1, x2, y2, padding=20
            )
            cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
            
            # Save cropped region
            crop_path = os.path.join(output_dir, f"02a_crop_{i+1}.jpg")
            cv2.imwrite(crop_path, cropped_bgr)
            print(f"      💾 Saved crop: {crop_path}")
            
            # Segment in cropped region
            print(f"      🎨 Segmenting object in crop...")
            cropped_mask = segment_cropped_region(cropped_rgb, sam2)
            
            # Save cropped mask
            mask_path = os.path.join(output_dir, f"02b_crop_mask_{i+1}.png")
            cv2.imwrite(mask_path, cropped_mask)
            print(f"      💾 Saved mask: {mask_path}")
            
            cropped_regions.append(cropped_bgr)
            cropped_masks.append(cropped_mask)
            crop_coords_list.append(crop_coords)
        
        print(f"\n   ✅ Processed {len(detection_boxes)} region(s)")
        
    except Exception as e:
        print(f"❌ Lỗi Segmentation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # =================================================================
    # BƯỚC 3: PASTE MASKS BACK
    # =================================================================
    print(f"\n{'='*70}")
    print("📋 BƯỚC 3: PASTE MASKS BACK TO ORIGINAL")
    print("="*70)
    
    # Create full-size mask
    full_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
    
    for i, (cropped_mask, crop_coords) in enumerate(zip(cropped_masks, crop_coords_list)):
        print(f"   Pasting mask {i+1}...")
        x1, y1, x2, y2 = crop_coords
        
        # Paste mask (use OR to combine multiple masks)
        temp_mask = paste_mask_to_original(
            original_image.shape,
            cropped_mask,
            crop_coords
        )
        full_mask = cv2.bitwise_or(full_mask, temp_mask)
    
    # Refine final mask
    print(f"\n   ✨ Refining final mask...")
    final_mask = refine_mask(full_mask, close_kernel=15, dilate_kernel=20)
    
    # Statistics
    mask_pixels = np.sum(final_mask > 0)
    coverage = (mask_pixels / (h_orig * w_orig)) * 100
    print(f"   ✅ Final mask: {mask_pixels:,} pixels ({coverage:.2f}%)")
    
    # Save final mask
    mask_path = os.path.join(output_dir, "03_final_mask.png")
    cv2.imwrite(mask_path, final_mask)
    print(f"   💾 Saved: {mask_path}")
    
    # =================================================================
    # BƯỚC 4: INPAINTING (LaMa)
    # =================================================================
    print(f"\n{'='*70}")
    print("🖌️  BƯỚC 4: INPAINTING (LaMa)")
    print("="*70)
    
    try:
        if not os.path.exists(lama_model_path):
            print(f"❌ Không tìm thấy LaMa tại: {lama_model_path}")
            return
        
        print("   📦 Loading LaMa...")
        inpainter = DeepInpaintingStrategy(model_path=lama_model_path)
        
        print("   ⏳ Đang xóa vật thể...")
        result_image = inpainter.process(original_image, final_mask)
        
        print("   ✅ Inpainting hoàn tất!")
        
        # Save result
        result_path = os.path.join(output_dir, "04_result.jpg")
        cv2.imwrite(result_path, result_image)
        print(f"   💾 Saved: {result_path}")
        
    except Exception as e:
        print(f"❌ Lỗi Inpainting: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # =================================================================
    # BƯỚC 5: VISUALIZATION
    # =================================================================
    print(f"\n{'='*70}")
    print("🎨 BƯỚC 5: CREATING VISUALIZATIONS")
    print("="*70)
    
    # Pipeline visualization
    pipeline_vis = visualize_pipeline_step(
        original_image,
        detection_boxes,
        cropped_regions,
        cropped_masks,
        final_mask
    )
    pipeline_path = os.path.join(output_dir, "05_pipeline.jpg")
    cv2.imwrite(pipeline_path, pipeline_vis)
    print(f"   💾 Pipeline: {pipeline_path}")
    
    # Final comparison
    h, w = original_image.shape[:2]
    mask_bgr = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
    combined = cv2.hconcat([original_image, mask_bgr, result_image])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, f"Prompt: {text_prompt}", (20, 40), font, 1, (0, 0, 255), 2)
    cv2.putText(combined, "Original", (20, h - 20), font, 0.8, (255, 255, 255), 2)
    cv2.putText(combined, "Mask", (w + 20, h - 20), font, 0.8, (255, 255, 255), 2)
    cv2.putText(combined, "Result", (w * 2 + 20, h - 20), font, 0.8, (255, 255, 255), 2)
    
    comparison_path = os.path.join(output_dir, "06_comparison.jpg")
    cv2.imwrite(comparison_path, combined)
    print(f"   💾 Comparison: {comparison_path}")
    
    # Display
    print("\n   📺 Displaying result...")
    window_name = "Smart Crop-Segment Pipeline - Press any key"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 600)
    cv2.imshow(window_name, combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # =================================================================
    # SUMMARY
    # =================================================================
    print("\n" + "="*70)
    print("✅ PIPELINE HOÀN THÀNH!")
    print("="*70)
    print(f"📁 Output: {output_dir}/")
    print(f"\n   Files created:")
    print(f"   - 02a_crop_*.jpg      : Cropped regions")
    print(f"   - 02b_crop_mask_*.png : Masks for each crop")
    print(f"   - 03_final_mask.png   : Combined final mask")
    print(f"   - 04_result.jpg       : Inpainted result")
    print(f"   - 05_pipeline.jpg     : Pipeline visualization")
    print(f"   - 06_comparison.jpg   : Before/after comparison")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()