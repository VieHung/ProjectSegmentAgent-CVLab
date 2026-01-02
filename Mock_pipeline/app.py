import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

# --- IMPORT MODULES CỦA BẠN (Giữ nguyên) ---
try:
    from modules.inpainting.strategies import TraditionalInpainting
    from modules.inpainting.deep_strategies import DeepInpaintingStrategy
    from modules.segmentation.intelligent_scissors import IntelligentScissorsApp
except ImportError:
    st.error("Lỗi: Không tìm thấy folder 'modules'. Hãy chạy lệnh streamlit tại thư mục chứa file main.py cũ.")
    st.stop()

# --- CẤU HÌNH ---
st.set_page_config(page_title="Inpainting Full Pipeline", layout="wide")
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TEMP_INPUT_PATH = "temp_input_image.jpg" # File tạm để OpenCV đọc được

# --- QUẢN LÝ STATE (Để lưu dữ liệu giữa các lần load trang) ---
if 'mask' not in st.session_state:
    st.session_state['mask'] = None
if 'processed_image' not in st.session_state:
    st.session_state['processed_image'] = None
if 'step' not in st.session_state:
    st.session_state['step'] = 1  # 1: Upload, 2: Mask, 3: Result

# --- HÀM LOAD MODEL ---
@st.cache_resource
def load_inpainter(use_ai):
    if use_ai:
        path = "weights/big-lama.pt"
        if os.path.exists(path):
            return DeepInpaintingStrategy(model_path=path)
        else:
            st.warning(f"⚠️ Không thấy {path}, chuyển về Cổ điển.")
            return TraditionalInpainting(method='ns', radius=3)
    else:
        return TraditionalInpainting(method='ns', radius=3)

def main():
    st.title("✂️ Intelligent Scissors Inpainting Tool")

    # --- CỘT TRÁI: CẤU HÌNH ---
    with st.sidebar:
        st.header("1. Upload Ảnh")
        uploaded_file = st.file_uploader("Chọn ảnh:", type=["jpg", "png"])
        
        st.header("2. Thuật toán")
        method = st.radio("Chọn Model:", ("AI (LaMa)", "Classic (Navier-Stokes)"))
        use_ai = True if method == "AI (LaMa)" else False

        # Nút Reset để làm lại từ đầu
        if st.button("🔄 Làm mới tất cả"):
            st.session_state['mask'] = None
            st.session_state['processed_image'] = None
            st.session_state['step'] = 1
            st.rerun()

    # --- LOGIC CHÍNH ---
    if uploaded_file is not None:
        # 1. Lưu file upload ra ổ cứng để IntelligentScissorsApp đọc được (Class này cần đường dẫn file)
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)
        # Convert RGB (PIL) -> BGR (OpenCV)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        cv2.imwrite(TEMP_INPUT_PATH, img_bgr)

        # Hien thi anh goc
        st.subheader("Bước 1: Ảnh gốc")
        st.image(image, caption="Original Image", use_column_width=True)

        # --- BƯỚC 2: SEGMENTATION (Dùng cửa sổ Popup) ---
        st.divider()
        st.subheader("Bước 2: Tạo Mask (Intelligent Scissors)")
        
        col_btn, col_info = st.columns([1, 3])
        
        with col_btn:
            # Nút bấm để mở cửa sổ OpenCV
            if st.button("✂️ MỞ CỬA SỔ VẼ MASK"):
                with st.spinner("Đang mở cửa sổ OpenCV... Hãy nhìn xuống thanh Taskbar!"):
                    # === ĐOẠN NÀY GỌI CODE CŨ CỦA BẠN ===
                    try:
                        # Khởi tạo App cũ
                        seg_app = IntelligentScissorsApp(TEMP_INPUT_PATH)
                        print("Đang mở cửa sổ vẽ...")
                        seg_app.run() # Cửa sổ sẽ hiện lên tại đây
                        
                        # Khi tắt cửa sổ (ESC), code chạy tiếp dòng này
                        if hasattr(seg_app, 'global_mask') and seg_app.global_mask is not None:
                            st.session_state['mask'] = seg_app.global_mask.copy()
                            cv2.destroyAllWindows()
                            st.success("Đã lấy được Mask!")
                        else:
                            st.error("Chưa tạo được mask. Hãy thử lại.")
                    except Exception as e:
                        st.error(f"Lỗi khi mở OpenCV: {e}")

        with col_info:
            if st.session_state['mask'] is not None:
                st.image(st.session_state['mask'], caption="Mask đã tạo", width=300)
                
                # Lưu Mask (Output 1)
                mask_path = os.path.join(OUTPUT_DIR, "01_segmentation_mask.png")
                cv2.imwrite(mask_path, st.session_state['mask'])
                st.caption(f"💾 Đã lưu: {mask_path}")
            else:
                st.info("👈 Nhấn nút bên trái. Một cửa sổ rời sẽ hiện ra. Vẽ xong nhấn ESC để quay lại đây.")

        # --- BƯỚC 3: INPAINTING & KẾT QUẢ ---
        if st.session_state['mask'] is not None:
            st.divider()
            st.subheader("Bước 3: Kết quả Xử lý")
            
            if st.button("🚀 Chạy Inpainting Ngay"):
                with st.spinner("Đang xử lý..."):
                    inpainter = load_inpainter(use_ai)
                    
                    # Process
                    try:
                        mask = st.session_state['mask']
                        result_bgr = inpainter.process(img_bgr, mask)
                        
                        # Lưu vào session state
                        st.session_state['processed_image'] = result_bgr
                        
                        # Lưu Output 2
                        res_path = os.path.join(OUTPUT_DIR, "02_inpainted_result.png")
                        cv2.imwrite(res_path, result_bgr)
                        
                        # Tạo ảnh so sánh (Output 3) - Logic cũ của bạn
                        h, w = img_bgr.shape[:2]
                        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                        mask_bgr = cv2.resize(mask_bgr, (w, h))
                        result_resized = cv2.resize(result_bgr, (w, h))
                        
                        combined = cv2.hconcat([img_bgr, mask_bgr, result_resized])
                        
                        # Label
                        label = "AI (LaMa)" if use_ai else "Classic (NS)"
                        cv2.putText(combined, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        
                        comp_path = os.path.join(OUTPUT_DIR, "03_comparison.png")
                        cv2.imwrite(comp_path, combined)
                        
                    except Exception as e:
                        st.error(f"Lỗi xử lý: {e}")

            # Hiển thị kết quả cuối cùng nếu đã có
            if st.session_state['processed_image'] is not None:
                # Convert BGR -> RGB để hiển thị web
                final_rgb = cv2.cvtColor(st.session_state['processed_image'], cv2.COLOR_BGR2RGB)
                st.image(final_rgb, caption="Kết quả Inpainting", use_column_width=True)
                
                st.success(f"✅ Hoàn tất! Tất cả file đã lưu tại thư mục: {OUTPUT_DIR}/")
                st.info(f"Đã lưu file so sánh: {os.path.join(OUTPUT_DIR, '03_comparison.png')}")

if __name__ == "__main__":
    main()