import sys
import os
import cv2
import numpy as np

# =================================================================
# CẤU HÌNH ĐƯỜNG DẪN TỰ ĐỘNG (RELATIVE PATH)
# =================================================================
# 1. Lấy đường dẫn tuyệt đối của thư mục chứa file main này
# Dù bạn chạy script từ đâu, BASE_DIR luôn trỏ đúng về folder dự án
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Tính toán đường dẫn đến source code GroundingDINO bên trong dự án
# Giả định cấu trúc: Project/modules/grounding/GroundingDINO
gd_path = os.path.join(BASE_DIR, "modules", "grounding", "GroundingDINO")

# 3. Thêm vào sys.path nếu chưa có
if gd_path not in sys.path:
    sys.path.append(gd_path)

# =================================================================
# IMPORT MODULES
# =================================================================
try:
    from modules.grounding.groundingDINO import GroundingDINOStrategy
    from modules.inpainting.deep_strategies import DeepInpaintingStrategy
except ImportError as e:
    print(f"❌ Lỗi Import: {e}")
    print(f"👉 Vui lòng kiểm tra xem thư mục này có tồn tại không: {gd_path}")
    sys.exit(1)

def main():
    # =================================================================
    # 0. CẤU HÌNH INPUT/OUTPUT (DÙNG OS.PATH.JOIN)
    # =================================================================
    # Định nghĩa các đường dẫn tương đối dựa trên BASE_DIR
    img_path = os.path.join(BASE_DIR, "inputs", "test_image1.jpg")
    output_dir = os.path.join(BASE_DIR, "outputs")
    
    # Weights Config
    dino_config = os.path.join(BASE_DIR, "weights", "" "GroundingDINO_SwinB_cfg.py")
    dino_weights = os.path.join(BASE_DIR, "weights", "groundingdino_swinb_cogcoor.pth")
    lama_model_path = os.path.join(BASE_DIR, "weights", "big-lama.pt")

    # Prompt
    text_prompt = input("Nhap prompt:")  # Vật thể muốn xóa
    
    # Tạo thư mục output nếu chưa có
    os.makedirs(output_dir, exist_ok=True)

    print(f"🚀 Bắt đầu chương trình...")
    print(f"   - Root Dir: {BASE_DIR}")
    print(f"   - Input: {os.path.basename(img_path)}")

    # Kiểm tra file ảnh
    if not os.path.exists(img_path):
        print(f"❌ Lỗi: Không tìm thấy file tại {img_path}")
        print("👉 Hãy copy ảnh vào thư mục inputs/ hoặc sửa tên file trong code.")
        return

    # Load ảnh
    original_image = cv2.imread(img_path)

    # =================================================================
    # BƯỚC 1: SEGMENTATION (GroundingDINO)
    # =================================================================
    print("\n--- BƯỚC 1: SEGMENTATION (GroundingDINO) ---")
    
    try:
        # Kiểm tra file weights trước khi load để tránh lỗi khó hiểu
        if not os.path.exists(dino_config) or not os.path.exists(dino_weights):
            print("❌ Thiếu file cấu hình hoặc weights cho GroundingDINO trong thư mục weights/")
            return

        detector = GroundingDINOStrategy(
            config_path=dino_config, 
            weights_path=dino_weights, 
            device=None 
        )
        
        print(f"🔍 Đang tìm vật thể: '{text_prompt}'...")
        boxes, scores = detector.detect(original_image, text_prompt=text_prompt)

        if len(boxes) == 0:
            print("❌ Không tìm thấy vật thể nào! Thử đổi prompt hoặc giảm ngưỡng confidence.")
            return
        
        print(f"✅ Đã tìm thấy {len(boxes)} vật thể.")

        # Tạo Mask
        mask = detector.create_mask_from_boxes(original_image.shape, boxes)
        
        # Lưu mask
        mask_output_path = os.path.join(output_dir, "01_dino_mask.png")
        cv2.imwrite(mask_output_path, mask)
        print(f"💾 Đã lưu Mask: {mask_output_path}")

    except Exception as e:
        print(f"❌ Lỗi Segment: {e}")
        return

    # =================================================================
    # BƯỚC 2: INPAINTING (LaMa)
    # =================================================================
    print("\n--- BƯỚC 2: INPAINTING (LaMa) ---")
    
    try:
        if not os.path.exists(lama_model_path):
            print(f"❌ Không tìm thấy model LaMa tại: {lama_model_path}")
            return

        print(">>> Đang khởi tạo AI Model...")
        inpainter = DeepInpaintingStrategy(model_path=lama_model_path)
        
        print("⏳ Đang xóa vật thể...")
        result_image = inpainter.process(original_image, mask)
        print("✅ Inpainting hoàn tất!")
        
        # Lưu kết quả
        result_output_path = os.path.join(output_dir, "02_dino_inpainted_result.jpg")
        cv2.imwrite(result_output_path, result_image)
        print(f"💾 Đã lưu kết quả: {result_output_path}")

    except Exception as e:
        print(f"❌ Lỗi Inpaint: {e}")
        return

    # =================================================================
    # BƯỚC 3: HIỂN THỊ VÀ SO SÁNH
    # =================================================================
    print("\n--- BƯỚC 3: HIỂN THỊ KẾT QUẢ ---")

    # Xử lý ảnh để hiển thị (Resize mask và result về đúng size gốc nếu cần)
    h, w = original_image.shape[:2]
    
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    if mask_bgr.shape[:2] != (h, w):
        mask_bgr = cv2.resize(mask_bgr, (w, h))
        
    if result_image.shape[:2] != (h, w):
        result_image = cv2.resize(result_image, (w, h))
    
    # Nối ảnh
    combined_result = cv2.hconcat([original_image, mask_bgr, result_image])

    # Vẽ chú thích
    cv2.putText(combined_result, f"Prompt: {text_prompt}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(combined_result, "Original", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(combined_result, "DINO Mask", (w + 20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(combined_result, "LaMa Result", (w * 2 + 20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Lưu ảnh so sánh
    comparison_path = os.path.join(output_dir, "03_dino_comparison.jpg")
    cv2.imwrite(comparison_path, combined_result)

    # Hiển thị cửa sổ
    window_name = "Project Segment Agent: Before vs After"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1400, 500) # Kích thước cửa sổ tùy chỉnh
    cv2.imshow(window_name, combined_result)
    
    print("\n✅ HOÀN THÀNH!")
    print(f"👉 File so sánh: {comparison_path}")
    print("👉 Nhấn phím bất kỳ để thoát.")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()