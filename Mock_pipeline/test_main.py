import os
import sys
import cv2
import numpy as np
import torch
import time

# 1. Lấy đường dẫn gốc
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Định nghĩa các đường dẫn cần thiết
# ĐƯỜNG DẪN QUAN TRỌNG NHẤT ĐỂ SỬA LỖI SAM2:
# Chúng ta cần thêm folder chứa chữ 'sam2', tức là folder 'segmentation'
segmentation_folder = os.path.join(BASE_DIR, "modules", "segmentation")

# Đường dẫn GroundingDINO
gd_folder = os.path.join(BASE_DIR, "modules", "grounding", "GroundingDINO")

# Đường dẫn Modules gốc (để import modules.segmentation...)
modules_root = os.path.join(BASE_DIR, "modules")

# 3. Thêm vào sys.path (Đưa lên đầu danh sách để ưu tiên tìm kiếm)
if segmentation_folder not in sys.path:
    sys.path.insert(0, segmentation_folder) # <--- QUAN TRỌNG: Insert vào đầu

if gd_folder not in sys.path:
    sys.path.insert(0, gd_folder)

if modules_root not in sys.path:
    sys.path.insert(0, modules_root)

# 4. Debug kiểm tra (Optional)
print(f"📂 Đã thêm path segmentation: {segmentation_folder}")
print(f"📂 Kiểm tra: {os.path.join(segmentation_folder, 'sam2', 'build_sam.py')}")

# --- BÂY GIỜ MỚI ĐƯỢC IMPORT ---
try:
    # Import class của bạn
    from modules.segmentation.sam2_mask_strategy import Sam2MaskStrategy
    from modules.grounding.groundingDINO import GroundingDINOStrategy
    from modules.inpainting.deep_strategies import DeepInpaintingStrategy
    print("✅ Import thành công!")
except ImportError as e:
    print(f"❌ Vẫn lỗi Import: {e}")
    # Mẹo debug: In ra sys.path để xem có đúng chưa
    import pprint
    pprint.pprint(sys.path)
    sys.exit(1)

import matplotlib
# Chuyển backend sang 'Agg' để tránh lỗi Segmentation Fault
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

class Visualizer:
    @staticmethod
    def show_comparison(image_bgr, mask, result_bgr, save_path="comparison_result.jpg", show_on_screen=False):
        """
        Hiển thị so sánh: Ảnh gốc - Mask - Kết quả
        Lưu ý: Đã tắt show_on_screen mặc định để tránh Segmentation Fault.
        """
        # 1. Chuyển đổi màu từ BGR (OpenCV) sang RGB (Matplotlib)
        # Convert sang RGB để hiển thị đúng màu
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        
        # 2. Tạo Overlay cho Mask
        mask_overlay = image_rgb.copy()
        mask_indices = mask > 0
        mask_overlay[mask_indices] = [255, 0, 0] # Tô đỏ
        alpha = 0.6
        overlay_viz = cv2.addWeighted(mask_overlay, alpha, image_rgb, 1 - alpha, 0)

        # 3. Vẽ biểu đồ
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(overlay_viz)
        axes[0].set_title("Original + Mask Overlay")
        axes[0].axis('off')

        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title("Dilated Mask (Input for LaMa)")
        axes[1].axis('off')

        axes[2].imshow(result_rgb)
        axes[2].set_title("Inpainted Result")
        axes[2].axis('off')

        # 4. Lưu ảnh
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"📊 Đã lưu ảnh so sánh chi tiết tại: {save_path}")
        
        # QUAN TRỌNG: Đóng figure để giải phóng RAM, tránh rò rỉ bộ nhớ
        plt.close(fig)

class ObjectRemoverAgent:
    """
    Agent quản lý luồng xóa vật thể:
    Text -> Box (DINO) -> Mask (SAM2) -> Dilate Mask -> Inpaint (LaMa)
    """
    def __init__(self, device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"\n🤖 [Agent] Khởi tạo Object Remover Agent trên {self.device}...")
        
        # 1. Khởi tạo GroundingDINO (Phát hiện vật)
        self.dino = GroundingDINOStrategy(
            config_path="weights/GroundingDINO_SwinB_cfg.py",
            weights_path="weights/groundingdino_swinb_cogcoor.pth",
            device=self.device
        )
        
        # 2. Khởi tạo SAM 2 (Tạo Mask chi tiết)
        self.sam2 = Sam2MaskStrategy(
            checkpoint_path="weights/sam2_hiera_base_plus.pt",
            config_path="modules/segmentation/configs/sam2/sam2_hiera_b+.yaml",
            device=self.device
        )
        
        # 3. Khởi tạo LaMa (Xóa vật & Tái tạo nền)
        self.lama = DeepInpaintingStrategy(
            model_path="weights/big-lama.pt",
            device=self.device
        )
        print("✅ [Agent] Sẵn sàng hoạt động!\n")

    def dilate_mask(self, mask, kernel_size=15):
        """
        Kỹ thuật mở rộng vùng mask để bao trọn viền vật thể.
        Giúp LaMa hoạt động tốt hơn, không bị lộ viền.
        """
        # Kernel hình vuông kích thước kernel_size x kernel_size
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Thực hiện phép toán Dilation (Nở vùng trắng)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        return dilated_mask

    def run(self, image_path, text_prompt, output_path="output_result.jpg", box_threshold=0.35):
        print(f"🖼️ Đang xử lý ảnh: {image_path} | Prompt: '{text_prompt}'")
        start_time = time.time()

        # 1. Đọc ảnh
        if not os.path.exists(image_path):
            print(f"❌ Không tìm thấy ảnh: {image_path}")
            return
        
        image = cv2.imread(image_path)
        if image is None:
            print("❌ Lỗi đọc ảnh.")
            return

        # --- BƯỚC 1: DETECTION ---
        print("🔍 Bước 1: Đang tìm vị trí vật thể (GroundingDINO)...")
        boxes, logits = self.dino.detect(image, text_prompt, box_threshold=box_threshold)
        
        if len(boxes) == 0:
            print(f"⚠️ Không tìm thấy đối tượng '{text_prompt}' nào trong ảnh.")
            return
        print(f"   -> Tìm thấy {len(boxes)} đối tượng.")

        # --- BƯỚC 2: SEGMENTATION ---
        print("✂️ Bước 2: Đang tách nền chi tiết (SAM 2)...")
        final_combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        for i, box in enumerate(boxes):
            single_box = np.array([box]) 
            mask = self.sam2.process(image, boxes=single_box)
            final_combined_mask = cv2.bitwise_or(final_combined_mask, mask)

        # --- BƯỚC 3: DILATION ---
        print("🎨 Bước 3: Đang mở rộng vùng mask (Dilation)...")
        # Dilate mask để LaMa hoạt động tốt hơn
        dilated_mask = self.dilate_mask(final_combined_mask, kernel_size=15)

        # --- BƯỚC 4: INPAINTING ---
        print("🖌️ Bước 4: Đang xóa vật thể và tái tạo nền (LaMa)...")
        result_image = self.lama.process(image, dilated_mask)

        # --- BƯỚC 5: LƯU & HIỂN THỊ KẾT QUẢ ---
        
        # Lưu kết quả cuối cùng
        cv2.imwrite(output_path, result_image)
        
        elapsed = time.time() - start_time
        print(f"✅ Hoàn tất! Ảnh đã lưu tại: {output_path}")
        print(f"⏱️ Tổng thời gian: {elapsed:.2f}s")
        
        # === PHẦN MỚI THÊM: GỌI VISUALIZER ===
        # Tạo tên file cho ảnh so sánh (vd: result_nomask_comparison.jpg)
        comp_path = output_path.replace(".jpg", "_comparison.jpg").replace(".png", "_comparison.png")
        
        Visualizer.show_comparison(
            image_bgr=image,
            mask=dilated_mask,
            result_bgr=result_image,
            save_path=comp_path,
            show_on_screen=True # Đặt False nếu chạy trên server không có GUI
        )

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Đảm bảo bạn đã tải weights về thư mục weights/
    # weights/groundingdino_swinb_cogcoor.pth
    # weights/sam2_hiera_base_plus.pt
    # weights/big-lama.pt
    
    # Cấu hình
    INPUT_IMAGE = "inputs/test_image1.jpg"  # Đường dẫn ảnh đầu vào
    PROMPT = "balloon"                # Vật thể cần xóa
    OUTPUT_IMAGE = "result_nomask.jpg"
    
    try:
        # Khởi tạo Agent
        agent = ObjectRemoverAgent()
        
        # Chạy
        # Tạo file dummy nếu chưa có để test code (bỏ qua nếu chạy thật)
        if not os.path.exists(INPUT_IMAGE):
            print(f"⚠️ Ví dụ: Không thấy {INPUT_IMAGE}, vui lòng thay bằng đường dẫn ảnh thật của bạn.")
        else:
            agent.run(INPUT_IMAGE, PROMPT, OUTPUT_IMAGE)
            
    except Exception as e:
        print(f"\n❌ Có lỗi xảy ra: {e}")
        import traceback
        traceback.print_exc()