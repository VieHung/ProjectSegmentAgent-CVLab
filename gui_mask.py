# File: gui_mask.py
import sys
import cv2
import os

# Import class vẽ của bạn
# Đảm bảo bạn đang đứng ở thư mục gốc khi chạy
try:
    from modules.segmentation.intelligent_scissors import IntelligentScissorsApp
except ImportError:
    # Fallback nếu chạy trực tiếp file này để test
    sys.path.append(os.getcwd())
    from modules.segmentation.intelligent_scissors import IntelligentScissorsApp

def run_gui(input_path, output_path):
    print(f"--- [GUI] Đang khởi động với ảnh: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"--- [GUI] Lỗi: Không tìm thấy file {input_path}")
        return

    # Khởi tạo App
    app = IntelligentScissorsApp(input_path)
    
    print("--- [GUI] Cửa sổ đã mở. Vui lòng vẽ...")
    app.run() # Cửa sổ sẽ treo ở đây chờ người dùng
    
    # Sau khi đóng cửa sổ
    if hasattr(app, 'global_mask') and app.global_mask is not None:
        print("--- [GUI] Đã nhận được mask. Đang lưu...")
        cv2.imwrite(output_path, app.global_mask)
        print(f"--- [GUI] Đã lưu mask thành công tại: {output_path}")
    else:
        print("--- [GUI] Không có mask nào được tạo (Người dùng hủy).")

if __name__ == "__main__":
    # Lấy tham số từ dòng lệnh: python gui_mask.py <input> <output>
    if len(sys.argv) < 3:
        print("Usage: python gui_mask.py <input_image_path> <output_mask_path>")
    else:
        in_path = sys.argv[1]
        out_path = sys.argv[2]
        run_gui(in_path, out_path)