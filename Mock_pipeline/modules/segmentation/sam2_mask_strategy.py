import torch
import numpy as np
import os
import cv2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class Sam2MaskStrategy:
    """
    Nhiệm vụ: Tạo Mask cực chuẩn từ Box hoặc Point (Không phải để xóa vật thể).
    Yêu cầu: 
        - File checkpoint (.pt): ví dụ sam2_hiera_large.pt
        - File config (.yaml): Tương ứng với model (nằm trong repo SAM2)
    """
    def __init__(self, 
                 checkpoint_path="weights/sam2_hiera_base_plus.pt", 
                 config_path="modules/segmentation/sam2/sam2_hiera_b+.yaml", # Cần đúng config của file .pt
                 device=None):
        
        # 1. Cấu hình thiết bị
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"🚀 [Sam2Mask] Initializing SAM 2 on {self.device}...")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"❌ Không tìm thấy weights tại: {checkpoint_path}")

        try:
            # 2. Build Model từ Config và Checkpoint
            # SAM 2 khác LaMa, nó cần load kiến trúc từ file config yaml trước
            sam2_model = build_sam2(config_path, checkpoint_path, device=self.device)
            
            # 3. Khởi tạo Predictor (Wrapper giúp xử lý ảnh dễ hơn)
            self.predictor = SAM2ImagePredictor(sam2_model)
            print("✅ Load model SAM 2 thành công!")
            
        except Exception as e:
            raise RuntimeError(f"❌ Lỗi khi load SAM 2: {e}\n👉 Kiểm tra lại file .yaml config có khớp với file .pt không.")

    def process(self, image: np.ndarray, boxes=None, points=None, labels=None) -> np.ndarray:
        """
        Input:
            - image: Ảnh gốc (Numpy array RGB)
            - boxes: Bounding box [x1, y1, x2, y2] (tùy chọn)
            - points: Tọa độ điểm [[x, y]] (tùy chọn)
            - labels: Nhãn cho điểm (1: foreground, 0: background)
        Output:
            - final_mask: Mask nhị phân (0 và 255) chuẩn kích thước ảnh gốc.
        """
        # --- 1. SET IMAGE (Encode ảnh - Bước này tốn time nhất của SAM) ---
        # SAM 2 yêu cầu ảnh RGB, uint8
        if hasattr(self, 'current_image_shape') and self.current_image_shape == image.shape:
             # (Optional) Nếu ảnh không đổi thì không cần set lại để tối ưu, 
             # nhưng an toàn nhất là cứ set lại nếu dùng cho API rời rạc.
             pass
        
        self.predictor.set_image(image)

        # --- 2. PREDICT MASK ---
        # SAM 2 có thể nhận box hoặc point
        masks, scores, _ = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=boxes,
            multimask_output=False # Chỉ lấy 1 mask tốt nhất
        )

        # --- 3. POST-PROCESSING ---
        # masks trả về shape (1, H, W) -> lấy ra (H, W)
        best_mask = masks[0]
        
        # Chuyển về định dạng ảnh grayscale (0-255) để dùng cho LaMa
        final_mask_uint8 = (best_mask * 255).astype(np.uint8)

        return final_mask_uint8
