import torch
import cv2
import numpy as np
import os
from core.interfaces import InpaintingStrategy

class DeepInpaintingStrategy(InpaintingStrategy):
    """
    Success Case 2: Sử dụng LaMa (Large Mask Inpainting) - Model SOTA.
    Yêu cầu: File weights/big-lama.pt
    """
    def __init__(self, model_path="weights/big-lama.pt", device=None):
        # 1. Cấu hình thiết bị (Ưu tiên GPU nếu có)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"🚀 [DeepInpainting] Initializing LaMa model on {self.device}...")

        # 2. Load Model thật (TorchScript)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Không tìm thấy model tại: {model_path}\n👉 Hãy tải file big-lama.pt và bỏ vào folder weights/ !")
        
        try:
            # Load model dạng JIT (đã gói gọn kiến trúc + weight)
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()
            print("✅ Load model LaMa thành công!")
        except Exception as e:
            raise RuntimeError(f"❌ Lỗi khi load model big-lama.pt: {e}")

    def process(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Quy trình xử lý ảnh qua Deep Learning Model
        """
        # --- 1. PRE-PROCESSING (Chuẩn bị dữ liệu) ---
        # LaMa yêu cầu kích thước ảnh phải chia hết cho 8
        h, w = image.shape[:2]
        new_h = (h // 8) * 8
        new_w = (w // 8) * 8
        
        # Resize ảnh và mask
        img_resized = cv2.resize(image, (new_w, new_h))
        mask_resized = cv2.resize(mask, (new_w, new_h))

        # Chuẩn hóa về Tensor [0, 1] và format (Batch, Channel, Height, Width)
        # Ảnh: (H, W, 3) -> (3, H, W) -> Chia 255
        img_tensor = torch.from_numpy(img_resized).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # Mask: (H, W) -> (1, H, W) -> Chia 255 (LaMa hiểu: 1 là vùng cần xóa, 0 là nền)
        if len(mask_resized.shape) == 2:
            mask_tensor = torch.from_numpy(mask_resized).float() / 255.0
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).to(self.device) # (1, 1, H, W)
        else:
             # Nếu mask input đã là 3 kênh, lấy 1 kênh thôi
            mask_tensor = torch.from_numpy(mask_resized[:,:,0]).float() / 255.0
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).to(self.device)

        # Binarize mask (đảm bảo mask chỉ có 0 hoặc 1)
        mask_tensor = (mask_tensor > 0.5).float()

        # --- 2. INFERENCE (Chạy model) ---
        with torch.no_grad():
            # Model LaMa (bản JIT) thường nhận đầu vào là ảnh và mask
            # Một số phiên bản yêu cầu concat, nhưng bản big-lama.pt phổ biến chạy như sau:
            try:
                # Cách 1: Truyền rời (Image, Mask) - Phổ biến với Sanster/IOPaint export
                output_tensor = self.model(img_tensor, mask_tensor)
            except:
                # Cách 2: Nếu model yêu cầu concat (Input 4 kênh)
                input_tensor = torch.cat([img_tensor, mask_tensor], dim=1)
                output_tensor = self.model(input_tensor)

        # --- 3. POST-PROCESSING (Xử lý kết quả) ---
        # Lấy kết quả từ GPU về CPU -> Numpy
        output_np = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output_np = np.clip(output_np * 255, 0, 255).astype(np.uint8)
        
        # Resize lại về kích thước gốc ban đầu của user
        output_final = cv2.resize(output_np, (w, h))
        
        # KỸ THUẬT BLENDING: Chỉ dán vùng được inpaint vào ảnh gốc
        # (Giữ nguyên nền gốc sắc nét, chỉ thay chỗ mask)
        mask_bool = mask > 0 # Vùng nào là mask thì lấy ảnh mới
        
        final_result = image.copy()
        # Lưu ý: output_final có thể hơi lệch màu một chút do model,
        # nhưng với LaMa thì thường rất khớp.
        final_result[mask_bool] = output_final[mask_bool]
        
        return final_result