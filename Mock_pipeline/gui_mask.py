import cv2
import sys
import numpy as np
import os

# Biến toàn cục
drawing = False
erasing = False
mask = None
brush_size = 15

# Biến lưu vị trí chuột cũ để vẽ nét liền mạch hơn
last_x, last_y = -1, -1

def draw_mask(event, x, y, flags, param):
    global drawing, erasing, mask, brush_size, last_x, last_y

    # --- NHẤN CHUỘT ---
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        last_x, last_y = x, y
        cv2.circle(mask, (x, y), brush_size, 255, -1)
        
    elif event == cv2.EVENT_RBUTTONDOWN:
        erasing = True
        last_x, last_y = x, y
        cv2.circle(mask, (x, y), brush_size, 0, -1)

    # --- DI CHUYỂN CHUỘT ---
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Vẽ đường thẳng từ điểm cũ đến điểm mới để nét không bị đứt đoạn khi di chuột nhanh
            cv2.line(mask, (last_x, last_y), (x, y), 255, brush_size * 2)
            cv2.circle(mask, (x, y), brush_size, 255, -1)
            last_x, last_y = x, y
        elif erasing:
            cv2.line(mask, (last_x, last_y), (x, y), 0, brush_size * 2)
            cv2.circle(mask, (x, y), brush_size, 0, -1)
            last_x, last_y = x, y

    # --- NHẢ CHUỘT ---
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
    elif event == cv2.EVENT_RBUTTONUP:
        erasing = False

def main():
    if len(sys.argv) < 3:
        print("Usage: python gui_mask.py <input_image> <output_mask>")
        return

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # 1. Đọc ảnh gốc
    img = cv2.imread(input_path)
    if img is None:
        print("Error: Cannot read input image")
        return

    h, w = img.shape[:2]

    # 2. Xử lý Mask (Load cũ hoặc tạo mới)
    global mask
    if os.path.exists(output_path):
        print(f"Loading existing mask from {output_path}")
        loaded_mask = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
        if loaded_mask is not None:
             mask = cv2.resize(loaded_mask, (w, h))
        else:
             mask = np.zeros((h, w), dtype=np.uint8)
    else:
        mask = np.zeros((h, w), dtype=np.uint8)

    # --- CẤU HÌNH CỬA SỔ (QUAN TRỌNG) ---
    window_name = 'Left: DRAW | Right: ERASE | S: SAVE | Q: QUIT'
    
    # Dùng WINDOW_NORMAL để cho phép resize thủ công
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) 

    # Tính toán kích thước hiển thị hợp lý (tối đa 1280x720 hoặc 80% màn hình)
    # Giúp ảnh to không bị tràn, ảnh nhỏ không bị bé quá
    screen_w, screen_h = 1280, 720 # Kích thước default an toàn
    
    scale_w = screen_w / w
    scale_h = screen_h / h
    scale = min(scale_w, scale_h)
    
    if scale < 1: # Chỉ thu nhỏ nếu ảnh lớn hơn màn hình
        new_w, new_h = int(w * scale), int(h * scale)
        cv2.resizeWindow(window_name, new_w, new_h)
    else:
        cv2.resizeWindow(window_name, w, h)

    cv2.setMouseCallback(window_name, draw_mask)

    print("--- GUIDE ---")
    print("Press '[' to decrease brush size")
    print("Press ']' to increase brush size")
    print("Press 'S' to Save & Exit")
    print("Press 'Q' to Quit without Saving")

    while True:
        # Tạo lớp phủ visual
        display_img = img.copy()
        
        # Tô màu đỏ lên vùng mask (kênh Red = 255)
        # Chỉ tô những chỗ mask màu trắng
        display_img[mask == 255] = [0, 0, 255] 
        
        # Blend: 70% ảnh gốc + 30% lớp phủ đỏ
        result = cv2.addWeighted(img, 0.7, display_img, 0.3, 0)

        # Hiển thị
        cv2.imshow(window_name, result)
        
        k = cv2.waitKey(1) & 0xFF
        
        if k == ord('s'): # Save
            cv2.imwrite(output_path, mask)
            print(f"Saved mask to {output_path}")
            break
        elif k == ord('q'): # Quit
            break
        elif k == ord('['): # Giảm brush
            brush_size = max(1, brush_size - 5)
            print(f"Brush size: {brush_size}")
        elif k == ord(']'): # Tăng brush
            brush_size = min(100, brush_size + 5)
            print(f"Brush size: {brush_size}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()