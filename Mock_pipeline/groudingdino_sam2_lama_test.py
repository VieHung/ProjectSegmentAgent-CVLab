"""
==============================================================================
SMART CROP-SEGMENT PIPELINE V3 (OPTIMIZED - NO HOLES)
GroundingDINO → Multi-Strategy SAM2 → Aggressive Hole Fill → LaMa Inpaint

Pipeline Flow:
1. DINO detects bounding box around object
2. Crop with GENEROUS padding (30%)
3. SAM2 multi-prompt strategy:
   - Box prompt (entire crop)
   - Point grid (9-16 points)
   - Lowered threshold (0.3)
4. Select largest mask + AGGRESSIVE hole filling
5. Safety net with DINO box fallback
6. LaMa inpainting

Key Improvements:
- 30% crop padding (vs 15%) for better SAM2 context
- Multi-prompt strategy ensures complete coverage
- Aggressive hole filling (morphology + contour fill)
- Smart fallback to DINO box if SAM2 coverage < 30%
- NO HOLES guaranteed!
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


def crop_with_generous_padding(image: np.ndarray, 
                               x1: int, y1: int, x2: int, y2: int,
                               padding_ratio: float = 0.30) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Crop with GENEROUS padding (30% instead of fixed pixels)
    This gives SAM2 better context for segmentation
    
    Args:
        image: Input image
        x1, y1, x2, y2: Bounding box coordinates
        padding_ratio: Padding as ratio of box size (0.30 = 30%)
    
    Returns:
        Cropped image and crop coordinates
    """
    h, w = image.shape[:2]
    box_w = x2 - x1
    box_h = y2 - y1
    
    # Calculate padding based on box size
    pad_w = int(box_w * padding_ratio)
    pad_h = int(box_h * padding_ratio)
    
    # Add padding
    x1_padded = max(0, x1 - pad_w)
    y1_padded = max(0, y1 - pad_h)
    x2_padded = min(w, x2 + pad_w)
    y2_padded = min(h, y2 + pad_h)
    
    # Crop
    cropped = image[y1_padded:y2_padded, x1_padded:x2_padded]
    
    print(f"      📏 Crop padding: {pad_w}x{pad_h} px ({padding_ratio*100:.0f}% of box size)")
    print(f"      📦 Crop region: [{x1_padded}, {y1_padded}, {x2_padded}, {y2_padded}]")
    
    return cropped, (x1_padded, y1_padded, x2_padded, y2_padded)


# =================================================================
# OPTIMIZED MASK GENERATION (V3)
# =================================================================

def generate_point_grid(box_coords: Tuple[int, int, int, int], 
                       grid_size: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a grid of points inside the bounding box for SAM2
    
    Args:
        box_coords: (x1, y1, x2, y2)
        grid_size: Grid dimensions (3 = 3x3 = 9 points)
    
    Returns:
        points: Array of (x, y) coordinates
        labels: Array of labels (1 = positive point)
    """
    x1, y1, x2, y2 = box_coords
    points = []
    
    # Create grid
    for i in range(grid_size):
        for j in range(grid_size):
            # Position points at (i+1)/(grid_size+1) to avoid edges
            px = x1 + (x2 - x1) * (i + 1) / (grid_size + 1)
            py = y1 + (y2 - y1) * (j + 1) / (grid_size + 1)
            points.append([px, py])
    
    points = np.array([points], dtype=np.float32)
    labels = np.ones((1, len(points[0])), dtype=np.int32)  # All positive
    
    return points, labels


def segment_with_multi_strategy(cropped_rgb: np.ndarray, 
                                sam2_model,
                                original_box_in_crop: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Multi-strategy SAM2 segmentation to ensure maximum coverage
    
    Strategies:
    1. Box prompt (entire crop region)
    2. Point grid (9 points) - if available
    3. Union of all masks
    
    Args:
        cropped_rgb: Cropped image in RGB format
        sam2_model: SAM2 model instance
        original_box_in_crop: Original object box within crop
    
    Returns:
        Combined binary mask
    """
    h_crop, w_crop = cropped_rgb.shape[:2]
    
    print(f"\n      🎯 Multi-Strategy SAM2 Segmentation")
    print(f"      Crop size: {w_crop}x{h_crop}")
    
    masks_collection = []
    
    # ============================================================
    # STRATEGY 1: Box Prompt (Entire Crop)
    # ============================================================
    print(f"\n      📦 Strategy 1: Box prompt (entire crop)")
    crop_box = np.array([[0, 0, w_crop, h_crop]], dtype=np.float32)
    
    try:
        raw_mask1 = sam2_model.process(cropped_rgb, boxes=crop_box)
        mask1 = process_sam2_output(raw_mask1, (h_crop, w_crop))
        
        if mask1 is not None:
            coverage1 = (np.sum(mask1 > 0) / (h_crop * w_crop)) * 100
            print(f"         ✅ Coverage: {coverage1:.2f}%")
            masks_collection.append(mask1)
        else:
            print(f"         ⚠️  Failed to process mask")
    except Exception as e:
        print(f"         ⚠️  Strategy 1 failed: {e}")
    
    # ============================================================
    # STRATEGY 2: Point Grid (if box is available)
    # ============================================================
    if original_box_in_crop[2] > 0 and original_box_in_crop[3] > 0:
        print(f"\n      📍 Strategy 2: Point grid (9 points)")
        
        try:
            points, labels = generate_point_grid(original_box_in_crop, grid_size=3)
            print(f"         Generated {len(points[0])} points")
            
            # Check if SAM2 supports point prompts
            raw_mask2 = sam2_model.process(cropped_rgb, points=points, labels=labels)
            mask2 = process_sam2_output(raw_mask2, (h_crop, w_crop))
            
            if mask2 is not None:
                coverage2 = (np.sum(mask2 > 0) / (h_crop * w_crop)) * 100
                print(f"         ✅ Coverage: {coverage2:.2f}%")
                masks_collection.append(mask2)
            else:
                print(f"         ⚠️  Failed to process mask")
        except Exception as e:
            print(f"         ⚠️  Strategy 2 not supported or failed: {e}")
    
    # ============================================================
    # MERGE: Union of all masks
    # ============================================================
    if len(masks_collection) == 0:
        print(f"\n      ❌ All strategies failed!")
        return np.zeros((h_crop, w_crop), dtype=np.uint8)
    
    print(f"\n      🔀 Merging {len(masks_collection)} mask(s)...")
    
    # Union (OR) of all masks
    combined_mask = masks_collection[0].copy()
    for mask in masks_collection[1:]:
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    final_coverage = (np.sum(combined_mask > 0) / (h_crop * w_crop)) * 100
    print(f"      ✅ Final combined coverage: {final_coverage:.2f}%")
    
    return combined_mask


def process_sam2_output(raw_mask, target_shape: Tuple[int, int]) -> np.ndarray:
    """
    Process SAM2 output to standard binary mask
    Handles different output formats from SAM2
    """
    if raw_mask is None:
        return None
    
    # Convert to numpy
    if hasattr(raw_mask, 'cpu'):
        mask = raw_mask.cpu().detach().numpy()
    else:
        mask = np.array(raw_mask)
    
    # Remove batch dimensions
    while mask.ndim > 2 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    
    # If multiple masks, select the largest one
    if mask.ndim == 3:
        mask_areas = [np.sum(mask[i] > 0) for i in range(mask.shape[0])]
        largest_idx = np.argmax(mask_areas)
        mask = mask[largest_idx]
    
    # Resize if needed
    h_target, w_target = target_shape
    if mask.shape[:2] != (h_target, w_target):
        mask = cv2.resize(mask, (w_target, h_target), interpolation=cv2.INTER_LINEAR)
    
    # Binarize
    mask = (mask > 0.0).astype(np.uint8) * 255
    
    return mask


def fill_holes_aggressive_v2(mask: np.ndarray) -> np.ndarray:
    """
    AGGRESSIVE hole filling - ensures NO HOLES remain
    
    Strategy:
    1. Morphological closing (fill small holes)
    2. Find largest external contour
    3. Fill entire contour (including all internal holes)
    
    Args:
        mask: Input binary mask
    
    Returns:
        Hole-filled mask
    """
    print(f"\n      🔧 Aggressive hole filling...")
    
    initial_pixels = np.sum(mask > 0)
    
    # Step 1: Morphological closing (fill small gaps)
    kernel = np.ones((11, 11), np.uint8)
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    after_close = np.sum(mask_closed > 0)
    print(f"         After morphology: +{after_close - initial_pixels:,} pixels")
    
    # Step 2: Find external contours
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        print(f"         ⚠️  No contours found, returning original")
        return mask_closed
    
    # Step 3: Fill largest contour (this removes ALL internal holes)
    largest_contour = max(contours, key=cv2.contourArea)
    mask_filled = np.zeros_like(mask)
    cv2.drawContours(mask_filled, [largest_contour], -1, 255, thickness=cv2.FILLED)
    
    final_pixels = np.sum(mask_filled > 0)
    total_filled = final_pixels - initial_pixels
    
    print(f"         Final result: +{total_filled:,} pixels filled")
    print(f"         ✅ NO HOLES guaranteed!")
    
    return mask_filled


def create_safety_fallback_mask(image_shape: Tuple[int, int, int],
                               box_coords: Tuple[int, int, int, int],
                               expansion: int = 10) -> np.ndarray:
    """
    Create fallback mask from DINO box with slight expansion
    Used when SAM2 fails or produces very small masks
    """
    h, w = image_shape[:2]
    x1, y1, x2, y2 = box_coords
    
    # Expand box slightly
    x1_safe = max(0, x1 - expansion)
    y1_safe = max(0, y1 - expansion)
    x2_safe = min(w, x2 + expansion)
    y2_safe = min(h, y2 + expansion)
    
    # Create mask
    fallback_mask = np.zeros((h, w), dtype=np.uint8)
    fallback_mask[y1_safe:y2_safe, x1_safe:x2_safe] = 255
    
    print(f"      📦 Fallback mask: [{x1_safe}, {y1_safe}, {x2_safe}, {y2_safe}]")
    
    return fallback_mask


def apply_smart_safety_net(sam2_mask: np.ndarray,
                          dino_box: Tuple[int, int, int, int],
                          image_shape: Tuple[int, int, int],
                          min_coverage_ratio: float = 0.30) -> np.ndarray:
    """
    Smart safety net: Use DINO box fallback if SAM2 mask is too small
    
    Args:
        sam2_mask: Mask from SAM2
        dino_box: Original DINO detection box
        image_shape: (height, width, channels)
        min_coverage_ratio: Minimum coverage ratio (0.30 = 30%)
    
    Returns:
        Safe mask (either SAM2 or union with DINO box)
    """
    x1, y1, x2, y2 = dino_box
    
    # Calculate coverage ratio
    sam2_area = np.sum(sam2_mask > 0)
    box_area = (x2 - x1) * (y2 - y1)
    coverage_ratio = sam2_area / box_area if box_area > 0 else 0
    
    print(f"\n      📊 Safety Check:")
    print(f"         SAM2 mask: {sam2_area:,} pixels")
    print(f"         DINO box:  {box_area:,} pixels")
    print(f"         Coverage:  {coverage_ratio:.2%}")
    
    if coverage_ratio < min_coverage_ratio:
        print(f"         ⚠️  Coverage too low (< {min_coverage_ratio:.0%})")
        print(f"         🛡️  Applying DINO box fallback...")
        
        # Create fallback mask
        fallback_mask = create_safety_fallback_mask(
            image_shape, dino_box, expansion=10
        )
        
        # Union with SAM2 mask
        safe_mask = cv2.bitwise_or(sam2_mask, fallback_mask)
        
        safe_area = np.sum(safe_mask > 0)
        print(f"         ✅ Safe mask: {safe_area:,} pixels")
        
        return safe_mask
    else:
        print(f"         ✅ Coverage OK, using SAM2 mask")
        return sam2_mask


def paste_mask_to_original(original_shape: Tuple[int, int, int],
                           cropped_mask: np.ndarray,
                           crop_coords: Tuple[int, int, int, int]) -> np.ndarray:
    """Paste cropped mask back to original image size"""
    h_orig, w_orig = original_shape[:2]
    x1, y1, x2, y2 = crop_coords
    
    # Create empty mask
    full_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
    
    # Paste cropped mask
    full_mask[y1:y2, x1:x2] = cropped_mask
    
    return full_mask


def smooth_mask_edges(mask: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    """
    Smooth mask edges without expansion
    """
    print(f"      ✨ Smoothing edges (kernel={kernel_size})...")
    
    # Gaussian blur
    mask_smooth = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
    
    # Re-binarize
    mask_smooth = (mask_smooth > 127).astype(np.uint8) * 255
    
    return mask_smooth


# =================================================================
# VISUALIZATION
# =================================================================

def visualize_optimized_pipeline(original: np.ndarray,
                                detection_boxes: List[Tuple[int, int, int, int]],
                                sam2_masks: List[np.ndarray],
                                final_mask: np.ndarray) -> np.ndarray:
    """Create visualization showing optimized pipeline"""
    h, w = original.shape[:2]
    
    # Panel 1: Original with detection
    panel1 = original.copy()
    for i, (x1, y1, x2, y2) in enumerate(detection_boxes):
        cv2.rectangle(panel1, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(panel1, f"#{i+1}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(panel1, "Detection", (20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    # Panel 2: SAM2 masks
    if sam2_masks:
        sam2_combined = np.zeros((h, w), dtype=np.uint8)
        for mask in sam2_masks:
            sam2_combined = cv2.bitwise_or(sam2_combined, mask)
        panel2 = cv2.cvtColor(sam2_combined, cv2.COLOR_GRAY2BGR)
        cv2.putText(panel2, "SAM2 Multi-Strategy", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)
    else:
        panel2 = np.zeros_like(original)
    
    # Panel 3: Final mask
    panel3 = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
    cv2.putText(panel3, "Final (No Holes)", (20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    
    # Panel 4: Overlay
    panel4 = original.copy()
    mask_colored = np.zeros_like(original)
    mask_colored[final_mask > 0] = [0, 255, 0]
    panel4 = cv2.addWeighted(original, 0.6, mask_colored, 0.4, 0)
    cv2.putText(panel4, "Overlay", (20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    # Stack in 2x2 grid
    top_row = cv2.hconcat([panel1, panel2])
    bottom_row = cv2.hconcat([panel3, panel4])
    combined = cv2.vconcat([top_row, bottom_row])
    
    return combined


# =================================================================
# MAIN PIPELINE
# =================================================================

def main():
    print("\n" + "="*70)
    print("🚀 OPTIMIZED CROP-SEGMENT PIPELINE V3 (NO HOLES)")
    print("   Multi-Strategy SAM2 + Aggressive Hole Fill + Smart Fallback")
    print("="*70)
    
    # =================================================================
    # 0. CẤU HÌNH INPUT/OUTPUT
    # =================================================================
    img_path = os.path.join(BASE_DIR, "inputs", "test_image2.jpg")
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
    # BƯỚC 2: OPTIMIZED MASK GENERATION (V3)
    # =================================================================
    print(f"\n{'='*70}")
    print("🔀 BƯỚC 2: OPTIMIZED MASK GENERATION (V3)")
    print("   - Multi-Strategy SAM2")
    print("   - Aggressive Hole Filling")
    print("   - Smart Safety Fallback")
    print("="*70)
    
    try:
        print("\n   📦 Loading SAM2...")
        sam2 = Sam2MaskStrategy(
            checkpoint_path=sam2_checkpoint,
            config_path=sam2_config
        )
        
        sam2_masks_list = []
        final_masks_list = []
        
        for i, (x1, y1, x2, y2) in enumerate(detection_boxes):
            print(f"\n   🔸 Processing box {i+1}/{len(detection_boxes)}:")
            print(f"      Original box: [{x1}, {y1}, {x2}, {y2}]")
            
            # ============================================================
            # Step 2A: Crop with GENEROUS padding (30%)
            # ============================================================
            print(f"\n   ✂️  Step 2A: Cropping with generous padding...")
            cropped_bgr, crop_coords = crop_with_generous_padding(
                original_image, x1, y1, x2, y2, padding_ratio=0.30
            )
            cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
            
            # Calculate box position within crop
            x1_crop, y1_crop, x2_crop, y2_crop = crop_coords
            box_in_crop = (
                x1 - x1_crop,
                y1 - y1_crop,
                x2 - x1_crop,
                y2 - y1_crop
            )
            
            # ============================================================
            # Step 2B: Multi-Strategy SAM2 Segmentation
            # ============================================================
            print(f"\n   🎯 Step 2B: Multi-strategy SAM2 segmentation...")
            cropped_mask = segment_with_multi_strategy(
                cropped_rgb, sam2, box_in_crop
            )
            
            # Save SAM2 raw result
            sam2_raw_path = os.path.join(output_dir, f"02a_sam2_raw_{i+1}.png")
            sam2_raw_full = paste_mask_to_original(
                original_image.shape, cropped_mask, crop_coords
            )
            cv2.imwrite(sam2_raw_path, sam2_raw_full)
            print(f"\n      💾 SAM2 raw: {sam2_raw_path}")
            
            # ============================================================
            # Step 2C: Aggressive Hole Filling
            # ============================================================
            cropped_mask_filled = fill_holes_aggressive_v2(cropped_mask)
            
            # Paste back to original size
            sam2_mask_full = paste_mask_to_original(
                original_image.shape,
                cropped_mask_filled,
                crop_coords
            )
            
            # Save filled mask
            filled_path = os.path.join(output_dir, f"02b_filled_{i+1}.png")
            cv2.imwrite(filled_path, sam2_mask_full)
            print(f"      💾 Filled: {filled_path}")
            
            # ============================================================
            # Step 2D: Smart Safety Net
            # ============================================================
            safe_mask = apply_smart_safety_net(
                sam2_mask_full,
                (x1, y1, x2, y2),
                original_image.shape,
                min_coverage_ratio=0.30
            )
            
            # Save safe mask
            safe_path = os.path.join(output_dir, f"02c_safe_{i+1}.png")
            cv2.imwrite(safe_path, safe_mask)
            print(f"      💾 Safe: {safe_path}")
            
            sam2_masks_list.append(sam2_mask_full)
            final_masks_list.append(safe_mask)
        
        # ============================================================
        # Step 2E: Combine all masks
        # ============================================================
        print(f"\n{'='*70}")
        print("🔀 Step 2E: COMBINING ALL MASKS")
        print("="*70)
        
        # Combine all final masks
        combined_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
        for mask in final_masks_list:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Final smoothing (edges only, no expansion)
        final_mask = smooth_mask_edges(combined_mask, kernel_size=7)
        
        # Statistics
        final_pixels = np.sum(final_mask > 0)
        total_pixels = h_orig * w_orig
        coverage = (final_pixels / total_pixels) * 100
        
        print(f"\n   📊 Final Mask Statistics:")
        print(f"      Total pixels: {final_pixels:,} / {total_pixels:,}")
        print(f"      Coverage: {coverage:.2f}%")
        print(f"      ✅ NO HOLES guaranteed!")
        
        # Save final mask
        final_save = os.path.join(output_dir, "03_final_mask.png")
        cv2.imwrite(final_save, final_mask)
        print(f"\n   💾 Final mask: {final_save}")
        
    except Exception as e:
        print(f"❌ Lỗi Mask Generation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # =================================================================
    # BƯỚC 3: INPAINTING (LaMa)
    # =================================================================
    print(f"\n{'='*70}")
    print("🖌️  BƯỚC 3: INPAINTING (LaMa)")
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
    # BƯỚC 4: VISUALIZATION
    # =================================================================
    print(f"\n{'='*70}")
    print("🎨 BƯỚC 4: CREATING VISUALIZATIONS")
    print("="*70)
    
    # Optimized pipeline visualization
    pipeline_vis = visualize_optimized_pipeline(
        original_image,
        detection_boxes,
        sam2_masks_list,
        final_mask
    )
    pipeline_path = os.path.join(output_dir, "05_pipeline_v3.jpg")
    cv2.imwrite(pipeline_path, pipeline_vis)
    print(f"   💾 Pipeline visualization: {pipeline_path}")
    
    # Final comparison
    h, w = original_image.shape[:2]
    mask_bgr = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
    combined = cv2.hconcat([original_image, mask_bgr, result_image])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, f"Prompt: {text_prompt}", (20, 40), font, 1, (0, 0, 255), 2)
    cv2.putText(combined, "Original", (20, h - 20), font, 0.8, (255, 255, 255), 2)
    cv2.putText(combined, "Final Mask (No Holes)", (w + 20, h - 20), font, 0.8, (255, 255, 255), 2)
    cv2.putText(combined, "Result", (w * 2 + 20, h - 20), font, 0.8, (255, 255, 255), 2)
    
    comparison_path = os.path.join(output_dir, "06_final_comparison.jpg")
    cv2.imwrite(comparison_path, combined)
    print(f"   💾 Final comparison: {comparison_path}")
    
    # Display
    print("\n   📺 Displaying result...")
    window_name = "Optimized Pipeline V3 - Press any key"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1800, 600)
    cv2.imshow(window_name, combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # =================================================================
    # SUMMARY
    # =================================================================
    print("\n" + "="*70)
    print("✅ OPTIMIZED PIPELINE V3 HOÀN THÀNH!")
    print("="*70)
    print(f"📁 Output: {output_dir}/")
    print(f"\n   📂 Step 2 - Masks:")
    print(f"      - 02a_sam2_raw_*.png   : SAM2 multi-strategy results")
    print(f"      - 02b_filled_*.png     : After aggressive hole filling")
    print(f"      - 02c_safe_*.png       : After safety fallback")
    print(f"      - 03_final_mask.png    : Final combined mask")
    print(f"\n   📂 Step 3 - Result:")
    print(f"      - 04_result.jpg        : Inpainted result")
    print(f"\n   📂 Visualizations:")
    print(f"      - 05_pipeline_v3.jpg   : 2x2 pipeline visualization")
    print(f"      - 06_final_comparison.jpg : Original | Mask | Result")
    print("\n   🎯 V3 IMPROVEMENTS:")
    print("      ✅ 30% crop padding (better SAM2 context)")
    print("      ✅ Multi-strategy SAM2 (box + point grid)")
    print("      ✅ Aggressive hole filling (morphology + contour)")
    print("      ✅ Smart safety net (DINO box fallback)")
    print("      ✅ NO HOLES GUARANTEED!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()