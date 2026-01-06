# File: gui_mask.py
import sys
import os
import cv2
import numpy as np

# =================================================================
# CẤU HÌNH ĐƯỜNG DẪN IMPORT
# =================================================================
# Lấy đường dẫn thư mục gốc (nơi chứa file này)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Thêm đường dẫn root vào sys.path để Python tìm thấy folder 'modules'
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Import module Intelligent Scissors
try:
    # Giả định đường dẫn file: modules/segmentation/intelligent_scissors.py
    # Và tên class là IntelligentScissorsApp
    from modules.segmentation.intelligent_scissors import IntelligentScissorsApp
except ImportError as e:
    print(f"❌ [GUI Error] Không thể import IntelligentScissorsApp: {e}")
    print(f"👉 Vui lòng kiểm tra file: {os.path.join(BASE_DIR, 'modules', 'segmentation', 'intelligent_scissors.py')}")
    sys.exit(1)

def run_gui(input_path, output_path):
    print(f"\n🚀 [GUI] Khởi động Intelligent Scissors...")
    print(f"   - Input: {input_path}")
    print(f"   - Output: {output_path}")
    
    # 1. Kiểm tra file input
    if not os.path.exists(input_path):
        print(f"❌ [GUI] Lỗi: Không tìm thấy ảnh tại {input_path}")
        return

    # 2. Khởi tạo App
    try:
        # Khởi tạo ứng dụng với đường dẫn ảnh
        app = IntelligentScissorsApp(input_path)
        
        print("\n--- HƯỚNG DẪN SỬ DỤNG KÉO THÔNG MINH ---")
        print("👉 Click chuột trái: Thêm điểm neo (Anchor point).")
        print("👉 Di chuột: Đường bao sẽ tự động bám theo cạnh vật thể.")
        print("👉 Enter: Kết thúc và đóng vùng chọn (tạo Mask).")
        print("👉 ESC: Hủy bỏ.")
        print("----------------------------------------\n")

        # 3. Chạy App (Code sẽ dừng tại đây cho đến khi user đóng cửa sổ)
        app.run()
        
        # 4. Lưu kết quả
        # Giả định class IntelligentScissorsApp có thuộc tính 'mask' lưu kết quả cuối cùng
        if hasattr(app, 'mask') and app.mask is not None:
            # Đảm bảo mask là binary (0 và 255)
            mask_to_save = app.mask
            if len(mask_to_save.shape) > 2:
                mask_to_save = cv2.cvtColor(mask_to_save, cv2.COLOR_BGR2GRAY)
            
            # Lưu file
            cv2.imwrite(output_path, mask_to_save)
            print(f"✅ [GUI] Đã lưu Mask thành công tại: {output_path}")
        else:
            print("⚠️ [GUI] Không có mask nào được tạo (Có thể bạn đã nhấn ESC hoặc chưa nhấn Enter).")

    except Exception as e:
        print(f"❌ [GUI] Lỗi Runtime: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Nhận tham số từ dòng lệnh: python gui_mask.py <input> <output>
    if len(sys.argv) < 3:
        print("Usage: python gui_mask.py <input_image_path> <output_mask_path>")
    else:
        in_path = sys.argv[1]
        out_path = sys.argv[2]
        run_gui(in_path, out_path)