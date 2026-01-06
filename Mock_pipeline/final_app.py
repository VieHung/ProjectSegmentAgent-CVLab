import streamlit as st
import cv2
import numpy as np
import os
import sys
import torch
import subprocess
import gdown  # <--- Thêm thư viện này
from PIL import Image

# =================================================================
# 1. CẤU HÌNH HỆ THỐNG & ĐƯỜNG DẪN
# =================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights") # Thư mục chứa weights

# Tạo thư mục weights nếu chưa có
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# Setup path cho modules
segmentation_folder = os.path.join(BASE_DIR, "modules", "segmentation")
gd_folder = os.path.join(BASE_DIR, "modules", "grounding", "GroundingDINO")
modules_root = os.path.join(BASE_DIR, "modules")

for p in [segmentation_folder, gd_folder, modules_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Setup folder output và file tạm
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
TEMP_INPUT_PATH = os.path.join(BASE_DIR, "temp_input_image.jpg")
TEMP_MASK_PATH = os.path.join(BASE_DIR, "temp_mask_output.png")

# =================================================================
# 2. IMPORT MODULES
# =================================================================
try:
    from modules.segmentation.sam2_mask_strategy import Sam2MaskStrategy
    from modules.grounding.groundingDINO import GroundingDINOStrategy
    from modules.inpainting.deep_strategies import DeepInpaintingStrategy
except ImportError as e:
    st.error(f"❌ Lỗi Import: {e}")
    st.stop()

# =================================================================
# 3. HELPER FUNCTIONS & MODEL LOADING
# =================================================================

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def dilate_mask(mask, kernel_size=15):
    if kernel_size <= 0: return mask
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)

def cleanup_temp_files():
    """Xóa file tạm sau khi dùng xong"""
    for f in [TEMP_INPUT_PATH, TEMP_MASK_PATH]:
        if os.path.exists(f):
            try: os.remove(f)
            except: pass

# --- HÀM TẢI WEIGHTS TỰ ĐỘNG ---
@st.cache_resource
def download_required_weights():
    """Tải các file weights từ Google Drive nếu chưa tồn tại"""
    
    # Dictionary: {Tên file: Google Drive ID}
    files_to_download = {
        "big-lama.pt": "1-s2qeHMEO5acm26_u3SpZKr3UiEmy4KU",
        "GroundingDINO_SwinB_cfg.py": "1dFTFUjLYQOs2cM33Q7-CMguxXWY0VYq_",
        "groundingdino_swinb_cogcoor.pth": "1jCq35XXzZuFB_vZAe3muva54-6qs9E_D",
        "sam2_hiera_base_plus.pt": "1PFlgFWEiNXHYwoN6WDebfOhee3CprwuX",
        "sam2.1_hiera_base_plus.pt": "11PV-z39Cbl8xAtgjAItqNLUpryDj51Ue"
    }

    st.toast("Đang kiểm tra file weights...", icon="📦")
    
    for filename, gdrive_id in files_to_download.items():
        file_path = os.path.join(WEIGHTS_DIR, filename)
        
        if not os.path.exists(file_path):
            url = f'https://drive.google.com/uc?id={gdrive_id}'
            try:
                # Hiển thị thông báo nhỏ
                print(f"Downloading {filename}...")
                gdown.download(url, file_path, quiet=False)
            except Exception as e:
                st.error(f"Không tải được {filename}: {e}")
    
    return True

# Gọi hàm tải ngay khi khởi động app
download_required_weights()

@st.cache_resource
def load_dino_model():
    # Load Config và Weights từ folder weights (đã tải ở trên)
    config = os.path.join(WEIGHTS_DIR, "GroundingDINO_SwinB_cfg.py")
    weights = os.path.join(WEIGHTS_DIR, "groundingdino_swinb_cogcoor.pth")
    
    if not os.path.exists(config) or not os.path.exists(weights):
        st.error("Thiếu file config hoặc weights cho DINO!")
        return None
        
    return GroundingDINOStrategy(config_path=config, weights_path=weights, device=get_device())

@st.cache_resource
def load_sam2_model():
    # Lưu ý: Code đang dùng bản 'sam2_hiera_base_plus.pt'
    checkpoint = os.path.join(WEIGHTS_DIR, "sam2_hiera_base_plus.pt")
    
    # Config YAML của SAM2 thường nằm trong code repo, không phải tải về
    # Nếu repo của bạn thiếu file yaml này thì báo lỗi, nhưng mình giữ nguyên logic cũ của bạn
    config = os.path.join(BASE_DIR, "modules", "segmentation", "configs", "sam2", "sam2_hiera_b+.yaml")
    
    if not os.path.exists(checkpoint):
        st.error(f"Thiếu file weights SAM2: {checkpoint}")
        return None

    return Sam2MaskStrategy(checkpoint_path=checkpoint, config_path=config, device=get_device())

@st.cache_resource
def load_lama_model():
    model_path = os.path.join(WEIGHTS_DIR, "big-lama.pt")
    
    if not os.path.exists(model_path):
        st.error(f"Không tìm thấy model LaMa tại: {model_path}")
        return None
        
    return DeepInpaintingStrategy(model_path=model_path, device=get_device())

# =================================================================
# 4. GIAO DIỆN STREAMLIT
# =================================================================
st.set_page_config(page_title="AI Object Remover Pro", layout="wide", page_icon="✂️")

st.title("✂️ AI Object Remover: Auto & Manual")
st.markdown("**SAM2/DINO** và **Intelligent Scissors**.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Cấu hình")
    
    # CHỌN CHẾ ĐỘ
    mode = st.radio("Chế độ (Mode):", ("🤖 Tự động (Text Prompt)", "✍️ Thủ công (Vẽ Mask)"))

    st.divider()
    
    # Cấu hình theo chế độ
    if mode == "🤖 Tự động (Text Prompt)":
        st.subheader("1. Detection (DINO)")
        text_prompt = st.text_input("Vật thể cần xóa:", value="person", help="Ví dụ: dog, car, balloon")
        box_threshold = st.slider("Độ nhạy (Threshold):", 0.1, 0.9, 0.35)
    else:
        st.info("Chế độ thủ công sẽ mở cửa sổ riêng để bạn cắt đối tượng chính xác hơn.")

    st.subheader("2. Inpainting (LaMa)")
    dilate_kernel = st.slider("Mở rộng vùng xóa (Dilate):", 0, 50, 15)

    if st.button("🔄 Reset App"):
        st.session_state.clear()
        cleanup_temp_files()
        st.rerun()

# --- MAIN LOGIC ---
uploaded_file = st.file_uploader("📂 Bước 1: Upload ảnh", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 1. Xử lý ảnh đầu vào
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # State Management
    if 'processed_mask' not in st.session_state: st.session_state['processed_mask'] = None
    if 'final_result' not in st.session_state: st.session_state['final_result'] = None
    if 'current_file' not in st.session_state: st.session_state['current_file'] = ""

    # Reset nếu đổi ảnh
    if st.session_state['current_file'] != uploaded_file.name:
        st.session_state['processed_mask'] = None
        st.session_state['final_result'] = None
        st.session_state['current_file'] = uploaded_file.name
        cleanup_temp_files()

    # Layout
    col1, col2, col3 = st.columns(3)

    # === CỘT 1: ẢNH GỐC & TẠO MASK ===
    with col1:
        st.subheader("🖼️ Ảnh Gốc")
        st.image(image_rgb, use_column_width=True)

        st.divider()
        st.write("### Bước 2: Tạo Mask")

        # LOGIC TẠO MASK DỰA TRÊN CHẾ ĐỘ
        if mode == "🤖 Tự động (Text Prompt)":
            if st.button("🔍 Tìm & Tạo Mask (AI)", type="primary"):
                try:
                    with st.spinner("Đang chạy DINO + SAM2..."):
                        dino = load_dino_model()
                        sam2 = load_sam2_model()
                        
                        if dino and sam2:
                            # Detect
                            boxes, _ = dino.detect(image_bgr, text_prompt, box_threshold=box_threshold)
                            
                            if len(boxes) == 0:
                                st.warning(f"Không tìm thấy '{text_prompt}'.")
                                st.session_state['processed_mask'] = None
                            else:
                                # Segment
                                combined_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
                                for box in boxes:
                                    m = sam2.process(image_bgr, boxes=np.array([box]))
                                    combined_mask = cv2.bitwise_or(combined_mask, m)
                                
                                st.session_state['processed_mask'] = combined_mask
                                st.success(f"Đã tìm thấy {len(boxes)} đối tượng.")

                except Exception as e:
                    st.error(f"Lỗi AI: {e}")

        else: # Chế độ Thủ công
            # Lưu ý: Chế độ thủ công dùng 'gui_mask.py' (cần GUI server, khó chạy trên Streamlit Cloud)
            # Trên Cloud, subprocess gọi GUI sẽ thất bại nếu không có X11 forwarding
            if st.button("✂️ Mở Cửa Sổ Vẽ Mask", type="primary"):
                # Lưu ảnh tạm để script con đọc
                cv2.imwrite(TEMP_INPUT_PATH, image_bgr)
                
                # Xóa mask cũ
                if os.path.exists(TEMP_MASK_PATH): os.remove(TEMP_MASK_PATH)
                st.session_state['processed_mask'] = None

                with st.spinner("Đang mở cửa sổ vẽ..."):
                    try:
                        # Gọi script gui_mask.py bằng subprocess
                        cmd = [sys.executable, "gui_mask.py", TEMP_INPUT_PATH, TEMP_MASK_PATH]
                        subprocess.run(cmd, check=True)
                        
                        # Kiểm tra kết quả
                        if os.path.exists(TEMP_MASK_PATH):
                            loaded_mask = cv2.imread(TEMP_MASK_PATH, cv2.IMREAD_GRAYSCALE)
                            if loaded_mask is not None:
                                # Resize mask về đúng size ảnh gốc
                                if loaded_mask.shape[:2] != image_bgr.shape[:2]:
                                    loaded_mask = cv2.resize(loaded_mask, (image_bgr.shape[1], image_bgr.shape[0]))
                                
                                st.session_state['processed_mask'] = loaded_mask
                                st.success("✅ Đã lấy Mask từ cửa sổ vẽ!")
                                st.rerun() 
                            else:
                                st.error("File mask bị lỗi.")
                        else:
                            st.warning("⚠️ Bạn đã đóng cửa sổ mà không lưu mask hoặc script lỗi.")
                    except subprocess.CalledProcessError as e:
                        st.error(f"Lỗi chạy gui_mask.py (Chế độ này chỉ chạy tốt ở Local): {e}")

    # === CỘT 2: KIỂM TRA MASK & INPAINT ===
    with col2:
        st.subheader("🎭 Mask (Segmentation)")
        
        if st.session_state['processed_mask'] is not None:
            # 1. Hiển thị mask hiện tại
            st.image(st.session_state['processed_mask'], caption="Mask hiện tại", use_column_width=True, clamp=True)

            # 2. Nút Chỉnh sửa thủ công
            st.write("---")
            if st.button("✏️ Chỉnh sửa / Bỏ chọn vùng thừa"):
                cv2.imwrite(TEMP_INPUT_PATH, image_bgr)
                cv2.imwrite(TEMP_MASK_PATH, st.session_state['processed_mask'])
                
                st.info("Đang mở cửa sổ... Chuột Trái: Vẽ | Chuột Phải: Xóa. Nhấn 'S' để Lưu.")
                try:
                    cmd = [sys.executable, "gui_mask.py", TEMP_INPUT_PATH, TEMP_MASK_PATH]
                    subprocess.run(cmd, check=True)
                    
                    if os.path.exists(TEMP_MASK_PATH):
                        refined_mask = cv2.imread(TEMP_MASK_PATH, cv2.IMREAD_GRAYSCALE)
                        if refined_mask is not None:
                            if refined_mask.shape[:2] != image_bgr.shape[:2]:
                                refined_mask = cv2.resize(refined_mask, (image_bgr.shape[1], image_bgr.shape[0]))
                            
                            st.session_state['processed_mask'] = refined_mask
                            st.success("✅ Đã cập nhật Mask!")
                            st.rerun()
                except Exception as e:
                    st.error(f"Lỗi chỉnh sửa (Chỉ chạy Local): {e}")

            st.write("---")

            # 3. Nút Chạy Inpainting (LaMa)
            if st.button("🚀 Bước 3: Xóa Vật Thể (LaMa)", type="primary"):
                try:
                    with st.spinner("Đang chạy Inpainting..."):
                        lama = load_lama_model()
                        if lama:
                            final_mask_input = dilate_mask(st.session_state['processed_mask'], kernel_size=dilate_kernel)
                            
                            result = lama.process(image_bgr, final_mask_input)
                            st.session_state['final_result'] = result
                            
                            out_name = f"result_{mode[:3]}_{uploaded_file.name}"
                            cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), result)
                            st.success("Xong!")
                except Exception as e:
                    st.error(f"Lỗi Inpainting: {e}")
        else:
            st.info("Chưa có mask. Hãy thực hiện Bước 2.")

    # === CỘT 3: KẾT QUẢ ===
    with col3:
        st.subheader("✨ Kết quả")
        if st.session_state['final_result'] is not None:
            res_rgb = cv2.cvtColor(st.session_state['final_result'], cv2.COLOR_BGR2RGB)
            st.image(res_rgb, use_column_width=True)

            is_success, buffer = cv2.imencode(".jpg", st.session_state['final_result'])
            if is_success:
                st.download_button(
                    label="⬇️ Tải ảnh về",
                    data=buffer.tobytes(),
                    file_name=f"result_{uploaded_file.name}",
                    mime="image/jpeg"
                )

else:
    st.info("👈 Vui lòng upload ảnh để bắt đầu.")