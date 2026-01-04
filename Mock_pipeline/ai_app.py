import streamlit as st
import cv2
import numpy as np
import os
import sys
import subprocess
from PIL import Image

# =================================================================
# 1. CẤU HÌNH & IMPORT
# =================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
TEMP_INPUT_PATH = os.path.join(BASE_DIR, "temp_input.jpg")
TEMP_MASK_PATH = os.path.join(BASE_DIR, "temp_mask.png")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Thêm path modules
gd_path = os.path.join(BASE_DIR, "modules", "grounding", "GroundingDINO")
if gd_path not in sys.path:
    sys.path.append(gd_path)

# Import Modules
try:
    from modules.grounding.groundingDINO import GroundingDINOStrategy
    from modules.inpainting.deep_strategies import DeepInpaintingStrategy
except ImportError as e:
    st.error(f"❌ Lỗi Import: {e}")
    st.stop()

# =================================================================
# 2. HÀM LOAD MODEL (CACHE)
# =================================================================
@st.cache_resource
def load_grounding_dino():
    config_path = os.path.join(BASE_DIR, "weights", "GroundingDINO_SwinB_cfg.py")
    weights_path = os.path.join(BASE_DIR, "weights", "groundingdino_swinb_cogcoor.pth")
    if not os.path.exists(config_path) or not os.path.exists(weights_path):
        return None
    return GroundingDINOStrategy(config_path, weights_path, device="cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu")

@st.cache_resource
def load_lama_inpainter():
    model_path = os.path.join(BASE_DIR, "weights", "big-lama.pt")
    if not os.path.exists(model_path):
        return None
    return DeepInpaintingStrategy(model_path=model_path)

# =================================================================
# 3. GIAO DIỆN STREAMLIT
# =================================================================
st.set_page_config(page_title="AI & Manual Inpainting", layout="wide", page_icon="🎨")

st.title("🎨 Advanced Object Removal Tool")
st.markdown("Kết hợp sức mạnh của **AI Tự động** và **Chỉnh sửa Thủ công**.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("🛠️ Chọn Chế Độ")
    mode = st.radio("Phương pháp tạo Mask:", ["🤖 AI Auto (Nhập text)", "✍️ Manual (Vẽ tay)"])
    
    st.divider()
    if mode == "🤖 AI Auto (Nhập text)":
        st.subheader("Cấu hình AI")
        text_prompt = st.text_input("Vật thể (VD: dog, car...):", value="person")
        box_threshold = st.slider("Độ tin cậy (Box):", 0.1, 0.9, 0.35)
        text_threshold = st.slider("Độ nhạy Text:", 0.1, 0.9, 0.25)
    else:
        st.info("ℹ️ Chế độ vẽ tay sẽ mở một cửa sổ riêng. Hãy vẽ bao quanh vật thể và nhấn 'Space' hoặc 'Enter' để hoàn tất.")

# --- MAIN ---
uploaded_file = st.file_uploader("📂 Upload ảnh cần xử lý:", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 1. Xử lý ảnh đầu vào
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, 1) # BGR
    original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Lưu ảnh tạm cho mode Manual dùng
    cv2.imwrite(TEMP_INPUT_PATH, original_image)

    # Layout hiển thị
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("1. Ảnh Gốc")
        st.image(original_rgb, use_column_width=True)

    # Quản lý State
    if 'mask' not in st.session_state: st.session_state['mask'] = None
    if 'current_img' not in st.session_state or st.session_state['current_img'] != uploaded_file.name:
        st.session_state['mask'] = None
        st.session_state['current_img'] = uploaded_file.name
        # Xóa file mask tạm cũ
        if os.path.exists(TEMP_MASK_PATH): os.remove(TEMP_MASK_PATH)

    # ---------------------------------------------------------
    # BƯỚC 2: TẠO MASK (TÙY CHỌN MODE)
    # ---------------------------------------------------------
    with col2:
        st.subheader("2. Tạo Mask")
        
        # === MODE A: AI AUTOMATIC ===
        if mode == "🤖 AI Auto (Nhập text)":
            if st.button("🔍 AI Phát Hiện"):
                detector = load_grounding_dino()
                if detector:
                    with st.spinner(f"Đang tìm '{text_prompt}'..."):
                        boxes, _ = detector.detect(original_image, text_prompt=text_prompt, box_threshold=box_threshold, text_threshold=text_threshold)
                        if len(boxes) > 0:
                            mask = detector.create_mask_from_boxes(original_image.shape, boxes)
                            st.session_state['mask'] = mask
                            st.success(f"Tìm thấy {len(boxes)} đối tượng.")
                        else:
                            st.warning("Không tìm thấy đối tượng nào.")
                            st.session_state['mask'] = None

        # === MODE B: MANUAL DRAWING ===
        else: # Manual Mode
            st.write("Dùng 'Intelligent Scissors' để chọn vùng chính xác.")
            if st.button("✂️ Mở Cửa Sổ Vẽ"):
                # Xóa mask cũ trước khi vẽ mới
                if os.path.exists(TEMP_MASK_PATH): os.remove(TEMP_MASK_PATH)
                st.session_state['mask'] = None

                with st.spinner("Đang mở cửa sổ vẽ... Vui lòng kiểm tra thanh taskbar nếu cửa sổ bị ẩn."):
                    try:
                        # Gọi process con để tránh treo Streamlit
                        cmd = [sys.executable, "gui_mask.py", TEMP_INPUT_PATH, TEMP_MASK_PATH]
                        subprocess.run(cmd, check=True)
                        
                        # Kiểm tra kết quả sau khi đóng cửa sổ
                        if os.path.exists(TEMP_MASK_PATH):
                            loaded_mask = cv2.imread(TEMP_MASK_PATH, cv2.IMREAD_GRAYSCALE)
                            if loaded_mask is not None:
                                st.session_state['mask'] = loaded_mask
                                st.success("✅ Đã lấy mask từ cửa sổ vẽ!")
                                st.rerun() # Refresh lại để hiện mask ngay
                        else:
                            st.warning("⚠️ Bạn đã đóng cửa sổ mà không lưu mask.")
                    except Exception as e:
                        st.error(f"Lỗi khi chạy tool vẽ: {e}")

        # HIỂN THỊ MASK
        if st.session_state['mask'] is not None:
            st.image(st.session_state['mask'], caption="Mask Đã Tạo", use_column_width=True)

    # ---------------------------------------------------------
    # BƯỚC 3: INPAINTING (CHẠY CHUNG CHO CẢ 2 MODE)
    # ---------------------------------------------------------
    with col3:
        st.subheader("3. Kết Quả")
        
        if st.session_state['mask'] is not None:
            if st.button("✨ Xóa Vật Thể (Inpaint)"):
                inpainter = load_lama_inpainter()
                if inpainter:
                    with st.spinner("Đang xử lý..."):
                        # Đảm bảo kích thước khớp
                        mask = st.session_state['mask']
                        if mask.shape[:2] != original_image.shape[:2]:
                            mask = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]))

                        # Process
                        result = inpainter.process(original_image, mask)
                        
                        # Hiển thị
                        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Kết quả Inpainting", use_column_width=True)
                        
                        # Lưu ảnh kết quả
                        res_path = os.path.join(OUTPUT_DIR, f"result_{uploaded_file.name}")
                        cv2.imwrite(res_path, result)
                        st.success(f"Đã lưu tại: {res_path}")
                else:
                    st.error("Không tìm thấy model LaMa!")
        else:
            st.info("Vui lòng tạo Mask ở bước 2 trước.")

else:
    st.info("👈 Hãy upload ảnh bên trái để bắt đầu.")