import streamlit as st
import cv2
import numpy as np
import os
import subprocess # <--- Dùng cái này thay cho multiprocessing
import sys
from PIL import Image

# --- IMPORT MODULES ---
try:
    from modules.inpainting.strategies import TraditionalInpainting
    from modules.inpainting.deep_strategies import DeepInpaintingStrategy
except ImportError:
    st.error("Lỗi: Không tìm thấy folder 'modules'.")
    st.stop()

# --- CẤU HÌNH ---
st.set_page_config(page_title="Inpainting Full Pipeline", layout="wide")
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TEMP_INPUT_PATH = "temp_input_image.jpg"
TEMP_MASK_PATH = "temp_mask_output.png"

# --- STATE MANAGEMENT ---
if 'mask' not in st.session_state: st.session_state['mask'] = None
if 'processed_image' not in st.session_state: st.session_state['processed_image'] = None
if 'uploader_key' not in st.session_state: st.session_state['uploader_key'] = 0
if 'current_file_name' not in st.session_state: st.session_state['current_file_name'] = ""

@st.cache_resource
def load_inpainter(use_ai):
    if use_ai:
        path = "weights/big-lama.pt"
        return DeepInpaintingStrategy(model_path=path) if os.path.exists(path) else TraditionalInpainting(method='ns', radius=3)
    else:
        return TraditionalInpainting(method='ns', radius=3)

def reset_callback():
    st.session_state['mask'] = None
    st.session_state['processed_image'] = None
    st.session_state['current_file_name'] = ""
    st.session_state['uploader_key'] += 1
    # Xóa các file tạm
    for f in [TEMP_INPUT_PATH, TEMP_MASK_PATH]:
        if os.path.exists(f):
            try: os.remove(f)
            except: pass

def main():
    st.title("✂️ Intelligent Scissors Inpainting Tool")

    with st.sidebar:
        st.header("1. Upload Ảnh")
        uploaded_file = st.file_uploader("Chọn ảnh:", type=["jpg", "png"], key=f"uploader_{st.session_state['uploader_key']}")
        
        st.header("2. Thuật toán")
        method = st.radio("Chọn Model:", ("AI (LaMa)", "Classic (Navier-Stokes)"))
        use_ai = True if method == "AI (LaMa)" else False

        st.button("🔄 Làm mới tất cả", on_click=reset_callback)

    if uploaded_file is not None:
        # Tự động reset nếu đổi file
        if uploaded_file.name != st.session_state['current_file_name']:
            st.session_state['mask'] = None
            st.session_state['processed_image'] = None
            st.session_state['current_file_name'] = uploaded_file.name

        # Lưu file tạm để script con đọc
        image = Image.open(uploaded_file).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        cv2.imwrite(TEMP_INPUT_PATH, img_bgr)

        st.subheader("Bước 1: Ảnh gốc")
        st.image(image, caption="Original Image", use_column_width=True)

        st.divider()
        st.subheader("Bước 2: Tạo Mask (Intelligent Scissors)")
        
        col_btn, col_info = st.columns([1, 3])
        
        with col_btn:
            if st.button("✂️ MỞ CỬA SỔ VẼ MASK"):
                # Xóa file mask cũ nếu có
                if os.path.exists(TEMP_MASK_PATH):
                    os.remove(TEMP_MASK_PATH)
                
                # Reset state mask
                st.session_state['mask'] = None

                with st.spinner("Đang mở cửa sổ riêng biệt..."):
                    # === GỌI SCRIPT CON (GUI_MASK.PY) ===
                    # Cách này tạo ra một process hệ điều hành hoàn toàn mới
                    # Không chia sẻ bất kỳ bộ nhớ nào với Streamlit -> Fix lỗi Qt
                    try:
                        cmd = [sys.executable, "gui_mask.py", TEMP_INPUT_PATH, TEMP_MASK_PATH]
                        print(f"Executing: {' '.join(cmd)}")
                        
                        subprocess.run(cmd, check=True)
                        
                    except subprocess.CalledProcessError as e:
                        st.error(f"Lỗi khi chạy cửa sổ vẽ: {e}")
                
                # Sau khi script con chạy xong, kiểm tra xem có file mask sinh ra không
                if os.path.exists(TEMP_MASK_PATH):
                    loaded_mask = cv2.imread(TEMP_MASK_PATH, cv2.IMREAD_GRAYSCALE)
                    if loaded_mask is not None:
                        st.session_state['mask'] = loaded_mask
                        st.success("✅ Đã lấy mask!")
                        st.rerun()
                    else:
                        st.error("Lỗi: File mask bị lỗi.")
                else:
                    st.warning("⚠️ Bạn đã đóng cửa sổ mà không lưu mask.")

        with col_info:
            if st.session_state['mask'] is not None:
                st.image(st.session_state['mask'], caption="Mask đã tạo", width=300)
                # Lưu output 1
                cv2.imwrite(os.path.join(OUTPUT_DIR, "01_segmentation_mask.png"), st.session_state['mask'])

        # --- BƯỚC 3: INPAINTING ---
        if st.session_state['mask'] is not None:
            st.divider()
            st.subheader("Bước 3: Kết quả")
            if st.button("🚀 Chạy Inpainting Ngay"):
                with st.spinner("Đang xử lý..."):
                    try:
                        inpainter = load_inpainter(use_ai)
                        mask = st.session_state['mask']
                        
                        # Resize an toàn
                        if mask.shape[:2] != img_bgr.shape[:2]:
                            mask = cv2.resize(mask, (img_bgr.shape[1], img_bgr.shape[0]))
                            
                        res = inpainter.process(img_bgr, mask)
                        st.session_state['processed_image'] = res
                        
                        cv2.imwrite(os.path.join(OUTPUT_DIR, "02_inpainted_result.png"), res)
                    except Exception as e:
                        st.error(f"Lỗi Inpainting: {e}")

            if st.session_state['processed_image'] is not None:
                st.image(cv2.cvtColor(st.session_state['processed_image'], cv2.COLOR_BGR2RGB), caption="Kết quả", use_column_width=True)

if __name__ == "__main__":
    main()