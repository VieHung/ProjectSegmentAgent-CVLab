import streamlit as st
import cv2
import numpy as np
import os
import sys
import torch
import time

# =================================================================
# 1. CẤU HÌNH HỆ THỐNG & ĐƯỜNG DẪN (QUAN TRỌNG)
# =================================================================
# Lấy đường dẫn gốc
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Định nghĩa các folder modules
segmentation_folder = os.path.join(BASE_DIR, "modules", "segmentation")
gd_folder = os.path.join(BASE_DIR, "modules", "grounding", "GroundingDINO")
modules_root = os.path.join(BASE_DIR, "modules")

# Thêm vào sys.path (Ưu tiên segmentation lên đầu để fix lỗi SAM2)
if segmentation_folder not in sys.path:
    sys.path.insert(0, segmentation_folder)
if gd_folder not in sys.path:
    sys.path.insert(0, gd_folder)
if modules_root not in sys.path:
    sys.path.insert(0, modules_root)

# Tạo folder output nếu chưa có
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =================================================================
# 2. IMPORT MODULES (Sau khi đã setup sys.path)
# =================================================================
try:
    from modules.segmentation.sam2_mask_strategy import Sam2MaskStrategy
    from modules.grounding.groundingDINO import GroundingDINOStrategy
    from modules.inpainting.deep_strategies import DeepInpaintingStrategy
except ImportError as e:
    st.error(f"❌ Lỗi Import: {e}")
    st.code("Gợi ý: Kiểm tra lại cấu trúc thư mục 'modules' và file '__init__.py'")
    st.stop()

# =================================================================
# 3. HELPER FUNCTIONS & CACHING
# =================================================================

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def dilate_mask(mask, kernel_size=15):
    """Kỹ thuật mở rộng vùng mask để bao trọn viền vật thể (cho LaMa)"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    return dilated_mask

# --- Load Models (Dùng @st.cache_resource để chỉ load 1 lần) ---

@st.cache_resource
def load_dino_model():
    print("⏳ Đang load GroundingDINO...")
    config = os.path.join(BASE_DIR, "weights", "GroundingDINO_SwinB_cfg.py")
    weights = os.path.join(BASE_DIR, "weights", "groundingdino_swinb_cogcoor.pth")
    return GroundingDINOStrategy(config_path=config, weights_path=weights, device=get_device())

@st.cache_resource
def load_sam2_model():
    print("⏳ Đang load SAM 2...")
    checkpoint = os.path.join(BASE_DIR, "weights", "sam2_hiera_base_plus.pt")
    config = os.path.join(BASE_DIR, "modules", "segmentation", "configs", "sam2", "sam2_hiera_b+.yaml")
    return Sam2MaskStrategy(checkpoint_path=checkpoint, config_path=config, device=get_device())

@st.cache_resource
def load_lama_model():
    print("⏳ Đang load LaMa...")
    model_path = os.path.join(BASE_DIR, "weights", "big-lama.pt")
    return DeepInpaintingStrategy(model_path=model_path, device=get_device())

# =================================================================
# 4. GIAO DIỆN STREAMLIT
# =================================================================
st.set_page_config(page_title="AI Object Remover (SAM2 + LaMa)", layout="wide", page_icon="🪄")

st.title("🪄 AI Object Remover: DINO + SAM2 + LaMa")
st.markdown("Xóa vật thể thông minh bằng cách nhập văn bản.")

# --- SIDEBAR: Cấu hình ---
with st.sidebar:
    st.header("⚙️ Cấu hình")
    
    st.subheader("1. Detection (DINO)")
    text_prompt = st.text_input("Vật thể cần xóa:", value="balloon", help="Nhập tên tiếng Anh, ví dụ: dog, car, person")
    box_threshold = st.slider("Box Threshold:", 0.1, 0.9, 0.35)
    
    st.subheader("2. Segmentation (SAM2)")
    # Ở đây SAM2 chạy tự động dựa trên box, ít tham số cần chỉnh
    
    st.subheader("3. Inpainting (LaMa)")
    dilate_kernel = st.slider("Mở rộng Mask (Dilate):", 0, 50, 15, help="Tăng lên nếu viền vật thể chưa xóa sạch")

    st.divider()
    if st.button("🧹 Xóa Cache Model"):
        st.cache_resource.clear()
        st.success("Đã xóa cache. Hãy reload trang.")

# --- MAIN SECTION ---
uploaded_file = st.file_uploader("📂 Bước 1: Upload ảnh", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Đọc ảnh vào OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Quản lý State (Lưu mask và kết quả giữa các lần render)
    if 'processed_mask' not in st.session_state: st.session_state['processed_mask'] = None
    if 'final_result' not in st.session_state: st.session_state['final_result'] = None
    if 'current_file' not in st.session_state or st.session_state['current_file'] != uploaded_file.name:
        # Reset state nếu đổi ảnh mới
        st.session_state['processed_mask'] = None
        st.session_state['final_result'] = None
        st.session_state['current_file'] = uploaded_file.name

    # Layout 3 cột
    col1, col2, col3 = st.columns(3)

    # --- CỘT 1: ẢNH GỐC ---
    with col1:
        st.subheader("🖼️ Ảnh Gốc")
        st.image(image_rgb, use_column_width=True)
        
        # Nút Action
        if st.button("🔍 Bước 2: Tìm & Tạo Mask", type="primary"):
            try:
                with st.spinner("Đang chạy GroundingDINO & SAM2..."):
                    # 1. Load Models
                    dino = load_dino_model()
                    sam2 = load_sam2_model()

                    # 2. Detect (DINO)
                    boxes, logits = dino.detect(image_bgr, text_prompt, box_threshold=box_threshold)
                    
                    if len(boxes) == 0:
                        st.error(f"⚠️ Không tìm thấy '{text_prompt}' trong ảnh.")
                        st.session_state['processed_mask'] = None
                    else:
                        st.toast(f"✅ Tìm thấy {len(boxes)} đối tượng!", icon="🎯")
                        
                        # 3. Segment (SAM2)
                        combined_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
                        for box in boxes:
                            single_box = np.array([box])
                            # SAM2 process
                            mask = sam2.process(image_bgr, boxes=single_box)
                            combined_mask = cv2.bitwise_or(combined_mask, mask)
                        
                        st.session_state['processed_mask'] = combined_mask
            except Exception as e:
                st.error(f"Lỗi khi tạo mask: {e}")

    # --- CỘT 2: MASK ---
    # --- CỘT 2: MASK ---
    with col2:
        st.subheader("🎭 Mask (Segmentation)")
        if st.session_state['processed_mask'] is not None:
            # Áp dụng Dilation ngay khi hiển thị để user thấy mask thực tế sẽ đưa vào LaMa
            display_mask = dilate_mask(st.session_state['processed_mask'], kernel_size=dilate_kernel)
            
            # --- SỬA LỖI TẠI ĐÂY (Đã bỏ cmap='gray') ---
            st.image(display_mask, caption=f"Mask (Dilate: {dilate_kernel})", use_column_width=True, clamp=True)

            if st.button("🎨 Bước 3: Xóa Vật Thể (Inpaint)"):
                try:
                    with st.spinner("Đang chạy LaMa Inpainting..."):
                        # 1. Load Model
                        lama = load_lama_model()
                        
                        # 2. Process
                        # Mask đã dilated ở trên visual, giờ tính lại cho chắc
                        mask_input = dilate_mask(st.session_state['processed_mask'], kernel_size=dilate_kernel)
                        result_bgr = lama.process(image_bgr, mask_input)
                        
                        # 3. Save State
                        st.session_state['final_result'] = result_bgr
                        
                        # Save file
                        out_path = os.path.join(OUTPUT_DIR, f"result_{uploaded_file.name}")
                        cv2.imwrite(out_path, result_bgr)
                        st.success("Đã xử lý xong!")
                except Exception as e:
                    st.error(f"Lỗi khi Inpaint: {e}")
        else:
            st.info("Chưa có mask. Hãy bấm nút 'Tìm & Tạo Mask' bên trái.")

    # --- CỘT 3: KẾT QUẢ ---
    with col3:
        st.subheader("✨ Kết Quả")
        if st.session_state['final_result'] is not None:
            res_rgb = cv2.cvtColor(st.session_state['final_result'], cv2.COLOR_BGR2RGB)
            st.image(res_rgb, use_column_width=True)
            
            # Download Button
            is_success, buffer = cv2.imencode(".jpg", st.session_state['final_result'])
            if is_success:
                st.download_button(
                    label="⬇️ Tải ảnh về",
                    data=buffer.tobytes(),
                    file_name=f"result_{uploaded_file.name}",
                    mime="image/jpeg"
                )
        else:
            st.info("Kết quả sẽ hiện ở đây.")

else:
    st.info("👈 Vui lòng upload ảnh để bắt đầu.")

# Debug info (Optional)
# st.write(f"Device đang chạy: {get_device()}")