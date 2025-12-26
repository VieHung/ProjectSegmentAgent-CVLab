import cv2
import numpy as np
from core.interfaces import InpaintingStrategy

class TraditionalInpainting(InpaintingStrategy):
    """
    Success Case 1: Sử dụng thuật toán cổ điển (OpenCV).
    Cân bằng giữa tốc độ và chất lượng bằng Navier-Stokes.
    """
    def __init__(self, method='ns', radius=3):
        # method: 'ns' (Navier-Stokes) hoặc 'telea'
        self.radius = radius
        self.method = cv2.INPAINT_NS if method == 'ns' else cv2.INPAINT_TELEA

    def process(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # 1. Validate Input
        if image is None or mask is None:
            raise ValueError("Image and Mask cannot be None")
        
        # 2. Chuẩn hóa Mask về 1 kênh (Grayscale) nếu lỡ đầu vào là 3 kênh
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # 3. Thực hiện Inpainting
        # Inpainting trong OpenCV yêu cầu mask 8-bit
        mask = mask.astype(np.uint8)
        
        output = cv2.inpaint(image, mask, self.radius, self.method)
        return output