import os
import torch
import cv2
import numpy as np
import sys


# Lấy đường dẫn tuyệt đối của thư mục chứa file hiện tại (modules/grounding)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Thêm thư mục này vào sys.path để Python tìm thấy folder 'GroundingDINO' nằm cùng cấp
if current_dir not in sys.path:
    sys.path.append(current_dir)

from PIL import Image
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.util.inference import load_model, predict


class GroundingDINOStrategy:
    """
    Phiên bản dùng Official Repo (IDEA-Research) với file .pth
    """
    def __init__(self, 
                 config_path="weights/GroundingDINO_SwinB_cfg.py", 
                 weights_path="weights/groundingdino_swinb_cogcoor.pth", 
                 device=None):
        
        # 1. Cấu hình thiết bị
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"🚀 [GroundingDINO-Official] Initializing on {self.device}...")

        # 2. Kiểm tra file tồn tại
        if not os.path.exists(config_path):
             raise FileNotFoundError(f"❌ Thiếu file config: {config_path}")
        if not os.path.exists(weights_path):
             raise FileNotFoundError(f"❌ Thiếu file weights: {weights_path}")

        # 3. Load Model bằng hàm của thư viện gốc
        try:
            self.model = load_model(config_path, weights_path, device=self.device)
            print("✅ Load model .pth thành công!")
        except Exception as e:
            raise RuntimeError(f"❌ Lỗi load model: {e}")

    def transform_image(self, image_pil):
        """
        Hàm xử lý ảnh theo chuẩn của GroundingDINO
        """
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_tensor, _ = transform(image_pil, None)
        return image_tensor

    def detect(self, image: np.ndarray, text_prompt: str, box_threshold=0.35, text_threshold=0.25):
        """
        Input: Ảnh OpenCV (numpy) + Prompt
        Output: List boxes [x1, y1, x2, y2] (Pixel coordinates)
        """
        # --- 1. PRE-PROCESSING ---
        # Chuyển OpenCV (BGR) -> PIL (RGB)
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Transform ảnh sang Tensor (Chuẩn hóa)
        image_tensor = self.transform_image(image_pil)

        # --- 2. INFERENCE ---
        # Hàm predict của thư viện gốc
        boxes, logits, phrases = predict(
            model=self.model,
            image=image_tensor,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device
        )

        # --- 3. POST-PROCESSING (Quan trọng) ---
        # Output 'boxes' của thư viện gốc ở dạng: [cx, cy, w, h] (Center X, Center Y, Width, Height)
        # Và giá trị được CHUẨN HÓA về [0, 1].
        # Chúng ta cần chuyển về: [x1, y1, x2, y2] (Pixel thực tế)
        
        h_img, w_img = image.shape[:2]
        boxes_pixel = []

        # Chuyển từ Tensor về Numpy để tính toán
        boxes_np = boxes.cpu().numpy()

        for box in boxes_np:
            cx, cy, w, h = box
            
            # De-normalize (Nhân với kích thước ảnh)
            cx *= w_img
            cy *= h_img
            w *= w_img
            h *= h_img

            # Chuyển từ (Center, Size) -> (TopLeft, BottomRight)
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)

            boxes_pixel.append([x1, y1, x2, y2])

        print(f"🔎 Tìm thấy {len(boxes_pixel)} đối tượng '{text_prompt}'")
        return np.array(boxes_pixel), logits

    def create_mask_from_boxes(self, image_shape, boxes):
        # (Giữ nguyên như code cũ)
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for box in boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        return mask