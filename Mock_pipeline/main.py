import cv2
import sys
import os
import numpy as np

# Import các module đã viết
from modules.inpainting.strategies import TraditionalInpainting
# Import thêm class Deep Learning mới viết
from modules.inpainting.deep_strategies import DeepInpaintingStrategy
from modules.segmentation.mock_model import ColorBasedSegmentation

# --- HƯỚNG DẪN TÍCH HỢP SAU NÀY ---
# Khi teamate của bạn xong việc (ví dụ họ tạo class: AISegmentation trong file ai_seg.py)
# Bạn chỉ cần:
# 1. Import class của họ: `from modules.segmentation.ai_seg import AISegmentation`
# 2. Thay thế dòng khởi tạo `seg_model` bên dưới.
# ----------------------------------

def main():
    # 1. Cấu hình đường dẫn ảnh
    image_path = "inputs/test_image2.jpg" # Hãy đảm bảo bạn có ảnh này
    
    # Kiểm tra file tồn tại
    if not os.path.exists(image_path):
        print(f"Lỗi: Không tìm thấy file {image_path}. Hãy copy 1 ảnh có vật thể màu đỏ vào folder inputs/")
        # Tạo ảnh giả để demo nếu không có ảnh thật
        img = 255 * np.ones((300, 300, 3), dtype=np.uint8)
        cv2.circle(img, (150, 150), 50, (0, 0, 255), -1) # Vẽ hình tròn đỏ
        cv2.imwrite(image_path, img)
        print("Đã tạo ảnh mẫu test_image.jpg (Hình tròn đỏ trên nền trắng)")

    # Load ảnh
    original_image = cv2.imread(image_path)

    # =================================================================
    # BƯỚC 1: SEGMENTATION (Tạo Mask)
    # =================================================================
    # Hiện tại: Dùng Mock Model (Tìm màu đỏ) - Bạn có thể đổi 'red', 'green' tùy ảnh
    seg_model = ColorBasedSegmentation(color_range='yellow') 
    
    # [TODO]: SAU NÀY SỬA DÒNG TRÊN THÀNH:
    # seg_model = TeamateSegmentationModel(weights="path/to/weights")
    
    print("Đang chạy Segmentation...")
    mask = seg_model.get_mask(original_image)
    
    # Hiển thị Mask để kiểm tra (Debug)
    cv2.imshow("Debug: Generated Mask", mask)

    # =================================================================
    # BƯỚC 2: INPAINTING (Phần của bạn)
    # =================================================================
    
    # --- CẤU HÌNH LỰA CHỌN THUẬT TOÁN ---
    # use_ai = False  -> Chạy Success Case 1 (Cổ điển - OpenCV)
    # use_ai = True   -> Chạy Success Case 2 (Deep Learning - LaMa)
    use_ai = True 

    if use_ai:
        print(">>> Đang khởi tạo AI Model (Case 2: LaMa)...")
        # Đảm bảo bạn đã tải file big-lama.pt vào thư mục weights/
        try:
            inpainter = DeepInpaintingStrategy(model_path="weights/big-lama.pt")
        except Exception as e:
            print(f"Lỗi khởi tạo AI: {e}")
            print("Đang chuyển về thuật toán Cổ điển...")
            inpainter = TraditionalInpainting(method='ns', radius=3)
    else:
        print(">>> Đang sử dụng thuật toán Cổ điển (Case 1: Navier-Stokes)...")
        inpainter = TraditionalInpainting(method='ns', radius=3)
    
    print("Đang chạy Inpainting...")
    try:
        result_image = inpainter.process(original_image, mask)
    except Exception as e:
        print(f"Lỗi quá trình xử lý: {e}")
        return

    # =================================================================
    # BƯỚC 3: HIỂN THỊ KẾT QUẢ
    # =================================================================
    # Nối ảnh lại để so sánh: Gốc | Mask | Kết quả
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) # Đổi sang 3 kênh để nối
    
    # Resize để đảm bảo ghép được (phòng trường hợp size lệch 1-2 pixel)
    h, w = original_image.shape[:2]
    mask_bgr = cv2.resize(mask_bgr, (w, h))
    result_image = cv2.resize(result_image, (w, h))
    
    combined_result = cv2.hconcat([original_image, mask_bgr, result_image])

    # Thêm text để biết đang dùng model nào
    label = "AI (LaMa)" if use_ai else "Classic (NS)"
    cv2.putText(combined_result, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Project 2 Demo: Original | Mask | Removed", combined_result)
    print("Nhấn phím bất kỳ để thoát...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()