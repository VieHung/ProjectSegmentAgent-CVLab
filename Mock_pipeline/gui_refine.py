import cv2
import sys
import os
import numpy as np

# ==========================================
# TRẠNG THÁI TOÀN CỤC (GLOBAL STATE)
# ==========================================
drawing = False
is_eraser = False  # False = Brush (Vẽ), True = Eraser (Xóa)
brush_size = 20
last_x, last_y = -1, -1

# Màu hiển thị giao diện
COLOR_MASK_OVERLAY = (0, 0, 255) # Đỏ
COLOR_BRUSH_CURSOR = (0, 255, 0) # Xanh lá (Khi vẽ)
COLOR_ERASER_CURSOR = (255, 255, 255) # Trắng (Khi xóa)

def mask_refine_app(image_path, mask_path):
    global drawing, is_eraser, brush_size, last_x, last_y

    # 1. Load dữ liệu
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Lỗi: Không đọc được ảnh input")
        return

    # Load mask hoặc tạo mới
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    else:
        mask = np.zeros(img.shape[:2], dtype=np.uint8)

    # 2. Thiết lập cửa sổ có thể resize (WINDOW_NORMAL)
    window_name = "Refine Mask (B: Brush | E: Eraser | [: Smaller | ]: Bigger | S: Save)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Tính toán kích thước hiển thị ban đầu hợp lý (không quá 1200px chiều rộng)
    h, w = img.shape[:2]
    aspect_ratio = w / h
    target_width = min(w, 1200)
    target_height = int(target_width / aspect_ratio)
    cv2.resizeWindow(window_name, target_width, target_height)

    # 3. Callback chuột
    def mouse_callback(event, x, y, flags, param):
        global drawing, last_x, last_y, brush_size, is_eraser

        # Cập nhật vị trí chuột để vẽ trỏ chuột (cursor)
        last_x, last_y = x, y

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            color = 0 if is_eraser else 255
            cv2.circle(mask, (x, y), brush_size, color, -1)

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                color = 0 if is_eraser else 255
                # Vẽ line để nét liền mạch khi di chuột nhanh
                cv2.circle(mask, (x, y), brush_size, color, -1)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    cv2.setMouseCallback(window_name, mouse_callback)

    print(f"\n--- HƯỚNG DẪN SỬ DỤNG MỚI ---")
    print(f"🖱️  Chuột Trái: Vẽ/Xóa (Tùy chế độ)")
    print(f"⌨️  Phím 'B': Chuyển sang BRUSH (Thêm vùng chọn)")
    print(f"⌨️  Phím 'E': Chuyển sang ERASER (Xóa vùng chọn)")
    print(f"⌨️  Phím '[' và ']': Giảm/Tăng kích thước cọ")
    print(f"💾  Phím 'S': LƯU và Thoát")
    print(f"❌  Phím 'ESC': Thoát không lưu")
    print(f"-----------------------------\n")

    while True:
        # --- RENDER GIAO DIỆN ---
        
        # 1. Tạo lớp phủ mask màu đỏ lên ảnh gốc
        # Chuyển mask grayscale sang 3 kênh để merge
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Chỉ lấy vùng có mask (màu trắng > 0)
        # Tạo màu đỏ ở những chỗ mask trắng
        red_layer = np.zeros_like(img)
        red_layer[:] = COLOR_MASK_OVERLAY
        
        # Logic blend:
        # Ở đâu mask trắng -> Hiển thị (Ảnh gốc * 0.6 + Đỏ * 0.4)
        # Ở đâu mask đen -> Hiển thị Ảnh gốc
        
        # Tạo mask boolean
        mask_bool = mask > 0
        
        display_img = img.copy()
        # Áp dụng alpha blending thủ công cho vùng mask
        display_img[mask_bool] = cv2.addWeighted(img[mask_bool], 0.6, red_layer[mask_bool], 0.4, 0)

        # 2. Vẽ con trỏ chuột (Vòng tròn) để user biết kích thước cọ
        if last_x != -1 and last_y != -1:
            cursor_color = COLOR_ERASER_CURSOR if is_eraser else COLOR_BRUSH_CURSOR
            cv2.circle(display_img, (last_x, last_y), brush_size, cursor_color, 1)
            
            # Hiển thị tâm
            cv2.circle(display_img, (last_x, last_y), 1, cursor_color, -1)

        # 3. Hiển thị thông tin trạng thái text lên góc ảnh
        mode_text = "MODE: ERASER (Xoa)" if is_eraser else "MODE: BRUSH (Ve)"
        text_color = (0, 0, 0) if is_eraser else (0, 255, 0) # Đen hoặc Xanh lá
        
        # Vẽ nền cho text dễ đọc
        cv2.rectangle(display_img, (10, 10), (250, 70), (255, 255, 255), -1)
        cv2.putText(display_img, mode_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        cv2.putText(display_img, f"Size: {brush_size}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,50,50), 1)

        cv2.imshow(window_name, display_img)

        # --- XỬ LÝ PHÍM BẤM ---
        key = cv2.waitKey(10) & 0xFF

        if key == 27: # ESC
            print("⚠️ Đã hủy bỏ chỉnh sửa.")
            break
        elif key == ord('s'): # Save
            cv2.imwrite(mask_path, mask)
            print("✅ Đã lưu Mask đã sửa!")
            break
        elif key == ord('e'): # Eraser mode
            is_eraser = True
        elif key == ord('b'): # Brush mode
            is_eraser = False
        elif key == ord(']'): # Tăng size
            brush_size += 2
        elif key == ord('['): # Giảm size
            brush_size = max(1, brush_size - 2)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        # Fallback cho test local không qua Streamlit
        print("Usage: python gui_refine.py <img_path> <mask_path>")
    else:
        mask_refine_app(sys.argv[1], sys.argv[2])