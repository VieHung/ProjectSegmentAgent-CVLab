import cv2
import numpy as np
from core.interfaces import SegmentationModel

class ColorBasedSegmentation(SegmentationModel):
    """
    [MOCK CLASS] - ĐÂY LÀ PHẦN GIẢ LẬP MODEL CỦA TEAMATE.
    Mục đích: Tự động tạo mask dựa trên màu sắc để bạn test inpainting.
    Ví dụ: Tự động tìm vật thể màu Đỏ để xóa.
    """
    def __init__(self, color_range='red'):
        self.color_range = color_range

    def get_mask(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Cấu trúc dữ liệu: Mỗi màu là một LIST các khoảng (lower, upper)
        # Phải dùng List vì màu Đỏ có 2 khoảng, màu khác chỉ có 1
        color_configs = {
            'red': [
                (np.array([0, 70, 50]), np.array([10, 255, 255])),
                (np.array([170, 70, 50]), np.array([180, 255, 255]))
            ],
            'blue': [
                (np.array([100, 150, 0]), np.array([140, 255, 255]))
            ],
            # Thêm màu mới rất gọn:
            'green': [
                (np.array([35, 50, 50]), np.array([85, 255, 255]))
            ],
            'yellow': [
                (np.array([20, 100, 100]), np.array([30, 255, 255]))                
            ]
        }

        # Lấy config, nếu không có trả về list rỗng
        ranges = color_configs.get(self.color_range, [])
        
        # Tạo mask nền đen
        final_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Vòng lặp để xử lý (Generic Logic)
        for (lower, upper) in ranges:
            mask = cv2.inRange(hsv, lower, upper)
            final_mask = cv2.bitwise_or(final_mask, mask) # Cộng gộp mask

        # Hậu xử lý chung
        kernel = np.ones((5,5), np.uint8)
        final_mask = cv2.dilate(final_mask, kernel, iterations=2)
        
        return final_mask