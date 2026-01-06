import cv2
import numpy as np
import sys

class IntelligentScissorsApp:
    def __init__(self, image_path):
        self.img = cv2.imread(image_path)
        if self.img is None:
            print(f"Lỗi: Không thể đọc file '{image_path}'")
            sys.exit()
            
        self.display_img = self.img.copy()
        self.h, self.w = self.img.shape[:2]
        
        # --- MỚI: TẠO MASK TOÀN CỤC ---
        # Mask này lưu trữ tất cả các vùng đã được chọn (trắng = chọn, đen = nền)
        self.global_mask = np.zeros((self.h, self.w), dtype=np.uint8)

        # Khởi tạo công cụ Intelligent Scissors
        self.tool = cv2.segmentation.IntelligentScissorsMB()
        self.tool.setEdgeFeatureCannyParameters(32, 100)
        self.tool.setGradientMagnitudeMaxLimit(200)
        
        print("Đang tính toán đặc trưng ảnh... vui lòng đợi.")
        self.tool.applyImage(self.img)
        print("Sẵn sàng!")

        # --- CÁC BIẾN QUẢN LÝ TRẠNG THÁI VẼ ---
        self.is_started = False
        self.anchors = []           
        self.fixed_contours = []    
        self.current_contour = None 
        
        # Thiết lập cửa sổ
        cv2.namedWindow("Magnetic Lasso Tool", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Magnetic Lasso Tool", 1200, 800)
        cv2.setMouseCallback("Magnetic Lasso Tool", self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            new_point = (x, y)
            if not self.is_started:
                self.is_started = True
                self.anchors = [new_point]
                self.tool.buildMap(new_point)
                print(f"-> Bắt đầu tại: {new_point}")
            else:
                if self.current_contour is not None:
                    self.fixed_contours.append(self.current_contour)
                self.anchors.append(new_point)
                self.tool.buildMap(new_point) 
                print(f"-> Đã chốt điểm {len(self.anchors)}: {new_point}")

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.is_started:
                self.finish_drawing()

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_started:
                self.current_contour = self.tool.getContour((x, y))
                self.update_display()

    def undo_last_step(self):
        if not self.is_started or len(self.anchors) == 0:
            print("Không có gì để Undo!")
            return

        if len(self.anchors) == 1:
            self.is_started = False
            self.anchors = []
            self.fixed_contours = []
            self.current_contour = None
            self.update_display()
            return

        self.anchors.pop()
        if len(self.fixed_contours) > 0:
            self.fixed_contours.pop()

        last_anchor = self.anchors[-1]
        self.tool.buildMap(last_anchor)
        self.current_contour = None 
        self.update_display()

    def finish_drawing(self):
        """Kết thúc 1 vòng vẽ và lưu vào mask"""
        if self.current_contour is not None:
            self.fixed_contours.append(self.current_contour)
        
        if len(self.anchors) > 1:
            start_pt = self.anchors[0]
            end_pt = self.anchors[-1]
            closing_line = np.array([end_pt, start_pt], dtype=np.int32).reshape((-1, 1, 2))
            self.fixed_contours.append(closing_line)

            # --- MỚI: LƯU VÀO MASK ---
            # Gộp tất cả các đoạn dây thành 1 đa giác liền mạch
            if len(self.fixed_contours) > 0:
                # np.vstack giúp nối các array con lại với nhau theo chiều dọc
                full_contour = np.vstack(self.fixed_contours)
                
                # Tô vùng này màu trắng (255) lên mask toàn cục
                cv2.fillPoly(self.global_mask, [full_contour], 255)
                
                #=========================================================================
                # Thêm phần mở rộng mask (để đảm bảo ko còn pixel màu đỏ của quả bóng nữa)
                #=========================================================================
                kernel = np.ones((5,5), np.uint8)
                self.global_mask = cv2.dilate(self.global_mask, kernel, iterations=2)
                print("-> [DEBUG] Mask đã được làm to ra (Dilated) giống Mock Model.")
                # ========================================================
                
                print("-> Đã lưu vùng chọn vào Mask. (Nhấn 'X' để xóa vùng này)")

        # Reset trạng thái vẽ để vẽ hình mới (nhưng giữ nguyên global_mask)
        self.is_started = False
        self.current_contour = None
        self.anchors = []
        self.fixed_contours = [] # Reset contour tạm
        self.update_display()

    def delete_masked_area(self):
        """Xóa vùng đã chọn khỏi ảnh gốc bằng Inpainting"""
        # Kiểm tra xem mask có dữ liệu không (có điểm trắng nào không)
        if cv2.countNonZero(self.global_mask) == 0:
            print("Chưa có vùng nào được chọn để xóa!")
            return

        print("Đang xóa vùng chọn và tái tạo ảnh (Inpainting)...")
        
        # --- THUẬT TOÁN XÓA ---
        # Dùng cv2.inpaint để "lấp đầy" vùng mask dựa trên các pixel xung quanh
        # bán kính = 3, thuật toán INPAINT_TELEA (thường cho kết quả tốt)
        self.img = cv2.inpaint(self.img, self.global_mask, 3, cv2.INPAINT_TELEA)
        
        # Reset mask về đen sau khi xóa xong
        self.global_mask[:] = 0
        
        # Cập nhật lại công cụ Intelligent Scissors với ảnh mới
        # (Để các lần cắt sau nhận diện đúng trên nền ảnh mới)
        self.tool.applyImage(self.img)
        
        print("-> Đã xóa xong!")
        self.update_display()

    def save_mask_to_file(self):
        """Lưu mask ra file png"""
        if cv2.countNonZero(self.global_mask) > 0:
            cv2.imwrite("mask_result.png", self.global_mask)
            print("-> Đã lưu file 'mask_result.png'")
        else:
            print("Mask rỗng, không có gì để lưu.")

    def update_display(self):
        canvas = self.img.copy()
        
        # --- MỚI: HIỂN THỊ MASK ---
        # Tạo hiệu ứng lớp phủ màu đỏ nhạt lên các vùng đã được đưa vào Mask
        # Lấy kênh đỏ của vùng có mask = 255 tăng lên
        if cv2.countNonZero(self.global_mask) > 0:
            # Tạo overlay màu đỏ
            overlay = canvas.copy()
            overlay[self.global_mask == 255] = [0, 0, 255] # BGR -> Đỏ
            # Trộn overlay với ảnh gốc (alpha blending)
            cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)

        # 1. Vẽ các đường dây đang vẽ dở (chưa finish)
        if len(self.fixed_contours) > 0:
            cv2.polylines(canvas, self.fixed_contours, isClosed=False, color=(0, 255, 0), thickness=2)
            
        # 2. Vẽ đoạn đang preview
        if self.is_started and self.current_contour is not None:
            cv2.polylines(canvas, [self.current_contour], isClosed=False, color=(0, 165, 255), thickness=2)
            
        # 3. Vẽ các điểm neo
        for pt in self.anchors:
             cv2.circle(canvas, pt, 4, (0, 0, 255), -1)

        cv2.imshow("Magnetic Lasso Tool", canvas)

    def run(self):
        self.update_display()
        print("--- HƯỚNG DẪN ---")
        print("1. Chuột Trái: Thêm điểm neo.")
        print("2. Chuột Phải: Kết thúc 1 vùng chọn (nhưng chưa thoát).")
        print("3. ENTER: LƯU MASK VÀ THOÁT.")
        print("4. Backspace: Undo.")
        print("5. ESC: Thoát không lưu.")
        
        while True:
            key = cv2.waitKey(20) & 0xFF
            
            if key == 27: # ESC -> Thoát mà không lưu (hoặc lưu rỗng)
                print("Đã nhấn ESC. Hủy bỏ.")
                self.mask = None 
                break

            elif key == 13: # Enter -> Lưu và Thoát
                print("Đã nhấn Enter. Đang xử lý mask và thoát...")
                
                # 1. Nếu đang vẽ dở thì đóng vòng dây lại trước
                if self.is_started:
                    self.finish_drawing()
                
                # 2. Gán global_mask vào biến 'mask' để script bên ngoài đọc được
                self.mask = self.global_mask 

                # 3. [QUAN TRỌNG] Lệnh này giúp thoát khỏi vòng lặp while -> Đóng cửa sổ
                break 

            elif key == 8: # Backspace -> Undo
                self.undo_last_step()

            elif key == ord('x') or key == ord('X'): # Phím X -> Xóa vùng (Demo)
                self.delete_masked_area()
                
            elif key == ord('s') or key == ord('S'): # Phím S -> Lưu file ảnh (Option)
                self.save_mask_to_file()
        
        cv2.destroyAllWindows()