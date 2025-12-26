from abc import ABC, abstractmethod
import numpy as np

class InpaintingStrategy(ABC):
    """
    Quy ước cho mọi thuật toán Inpainting (của BẠN).
    Input: Ảnh gốc (H, W, 3), Mask (H, W) - Quy ước: 255 là vùng cần xóa, 0 là nền.
    Output: Ảnh đã xóa vật thể (H, W, 3).
    """
    @abstractmethod
    def process(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        pass

class SegmentationModel(ABC):
    """
    Quy ước cho model Segmentation (của TEAMATE).
    Input: Ảnh gốc.
    Output: Mask nhị phân (trắng là vật cần xóa).
    """
    @abstractmethod
    def get_mask(self, image: np.ndarray) -> np.ndarray:
        pass