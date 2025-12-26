import cv2
import sys
import os
import numpy as np

# Import các module đã viết
from modules.inpainting.strategies import TraditionalInpainting
# Import thêm class Deep Learning mới viết
from modules.inpainting.deep_strategies import DeepInpaintingStrategy

# --- THAY ĐỔI: Import Intelligent Scissors thay cho Mock Model ---
from modules.segmentation.intelligent_scissors import IntelligentScissorsApp

# --- HƯỚNG DẪN TÍCH HỢP SAU NÀY ---
# Khi teamate của bạn xong việc (ví dụ họ tạo class: AISegmentation trong file ai_seg.py)
# Bạn chỉ cần:
# 1. Import class của họ: `from modules.segmentation.ai_seg import AISegmentation`
# 2. Thay thế dòng khởi tạo `seg_model` bên dưới.
# ----------------------------------

def main():
    # 1. Cấu hình đường dẫn ảnh
    image_path = "inputs/test_image1.jpg" # Hãy đảm bảo bạn có ảnh này
    
    # Tạo thư mục outputs nếu chưa có
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
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
    # BƯỚC 1: SEGMENTATION (Tạo Mask) - ĐÃ SỬA ĐỔI
    # =================================================================
    
    # --- THAY ĐỔI: Dùng Intelligent Scissors thay cho ColorBasedSegmentation --- 
    # Khởi tạo Interactive Segmentation Tool
    seg_app = IntelligentScissorsApp(image_path)
    seg_app.update_display()
    print("\nĐang chạy Segmentation... Vẽ mask và nhấn ESC khi xong.")
    
    # Vòng lặp vẽ mask
    seg_app.run()
    
    # Lấy mask đã vẽ
    mask = seg_app.global_mask.copy()
    
    # Đóng cửa sổ Intelligent Scissors
    cv2.destroyAllWindows()
    
    # --- LƯU MASK (OUTPUT 1) ---
    mask_output_path = os.path.join(output_dir, "01_segmentation_mask.png")
    cv2.imwrite(mask_output_path, mask)
    print(f"💾 Đã lưu Mask: {mask_output_path}")
    
    # Hiển thị Mask để kiểm tra (Debug)
    cv2.imshow("Debug: Generated Mask", mask)
    print("→ Nhấn phím bất kỳ để tiếp tục sang bước Inpainting...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # =================================================================
    # BƯỚC 2: INPAINTING (Phần của bạn)
    # =================================================================
    
    print("\n" + "=" * 60)
    print("BƯỚC 2: INPAINTING")
    print("=" * 60)
    
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
        print("✓ Inpainting hoàn tất!")
        
        # --- LƯU ẢNH SAU INPAINTING (OUTPUT 2) ---
        result_output_path = os.path.join(output_dir, "02_inpainted_result.png")
        cv2.imwrite(result_output_path, result_image)
        print(f"💾 Đã lưu ảnh kết quả: {result_output_path}")
        
    except Exception as e:
        print(f"Lỗi quá trình xử lý: {e}")
        return

    # =================================================================
    # BƯỚC 3: HIỂN THỊ KẾT QUẢ
    # =================================================================
    print("\n" + "=" * 60)
    print("BƯỚC 3: HIỂN THỊ KẾT QUẢ")
    print("=" * 60)
    
    # Nối ảnh lại để so sánh: Gốc | Mask | Kết quả
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) # Đổi sang 3 kênh để nối
    
    # Resize để đảm bảo ghép được (phòng trường hợp size lệch 1-2 pixel)
    h, w = original_image.shape[:2]
    mask_bgr = cv2.resize(mask_bgr, (w, h))
    result_image = cv2.resize(result_image, (w, h))
    
    combined_result = cv2.hconcat([original_image, mask_bgr, result_image])
    
    # Thêm text label...
    label = "AI (LaMa)" if use_ai else "Classic (NS)"
    cv2.putText(combined_result, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # --- LƯU ẢNH ---
    comparison_output_path = os.path.join(output_dir, "03_comparison.png")
    cv2.imwrite(comparison_output_path, combined_result)
    print(f"💾 Đã lưu ảnh so sánh: {comparison_output_path}")

    # --- HIỂN THỊ (SỬ DỤNG CÁCH CỦA BẠN) ---
    window_name = "Project 2 Demo: Original | Mask | Removed" # Đặt tên biến để tránh gõ sai
    
    # 1. Tạo cửa sổ ở chế độ NORMAL (cho phép resize)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) 
    
    # 2. Thiết lập kích thước cửa sổ hiển thị (Ví dụ: 1200x600)
    # Lưu ý: Nên set tỷ lệ tương đương ảnh gốc để không bị méo hình
    cv2.resizeWindow(window_name, 1200, 600) 

    # 3. Hiển thị ảnh vào cửa sổ đó
    cv2.imshow(window_name, combined_result)

    print("Nhấn phím bất kỳ để thoát...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print("✓ HOÀN THÀNH!")
    print("=" * 60)
    print(f"📁 Tất cả file đã được lưu trong thư mục: {output_dir}/")
    print(f"   1. {mask_output_path}")
    print(f"   2. {result_output_path}")
    print(f"   3. {comparison_output_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()