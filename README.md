# Intelligent Scissors Inpainting Tool

A computer vision pipeline that combines interactive segmentation (Intelligent Scissors / Magnetic Lasso) with image inpainting to remove objects from images. Supports both a classic OpenCV-based inpainting algorithm and a deep learning approach using the LaMa model.

## Features

- **Interactive Segmentation** – Draw precise object boundaries using the Intelligent Scissors (Magnetic Lasso) tool powered by `cv2.segmentation.IntelligentScissorsMB`.
- **Classic Inpainting** – Remove selected regions using OpenCV's Navier-Stokes (`INPAINT_NS`) or TELEA algorithm (no extra weights required).
- **Deep Learning Inpainting** – Remove selected regions using the [LaMa](https://github.com/advimman/lama) (Large Mask Inpainting) model for higher-quality results.
- **Streamlit Web UI** – An interactive browser-based interface (`app.py`) for uploading images, drawing masks, and running inpainting.
- **Command-Line Interface** – A scriptable CLI pipeline (`main.py`) for automated processing.

## Project Structure

```
.
├── app.py                          # Streamlit web UI entry point
├── main.py                         # Command-line pipeline entry point
├── gui_mask.py                     # Standalone mask-drawing window (called by app.py)
├── requirements.txt
├── inputs/                         # Place input images here
├── outputs/                        # Processed results are saved here
│   ├── 01_segmentation_mask.png
│   ├── 02_inpainted_result.png
│   └── 03_comparison.png
├── weights/                        # Place model weights here (created manually)
│   └── big-lama.pt
├── core/
│   └── interfaces.py               # Abstract base classes (InpaintingStrategy, SegmentationModel)
└── modules/
    ├── inpainting/
    │   ├── strategies.py           # TraditionalInpainting (OpenCV)
    │   └── deep_strategies.py      # DeepInpaintingStrategy (LaMa)
    └── segmentation/
        └── intelligent_scissors.py # IntelligentScissorsApp
```

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/VieHung/ProjectSegmentAgent-CVLab.git
   cd ProjectSegmentAgent-CVLab
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Download LaMa weights for deep learning inpainting**

   Create a `weights/` directory and download the pre-trained model:

   ```bash
   mkdir weights
   # Download big-lama.pt from:
   # https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt
   # and place it inside the weights/ folder.
   ```

## Usage

### Option 1 – Command-Line Interface (`main.py`)

Place an image named `test_image1.jpg` in the `inputs/` folder, then run:

```bash
python main.py
```

The pipeline will:
1. Open an **Intelligent Scissors** window – draw your mask and press **ESC** when done.
2. Run inpainting (LaMa by default; falls back to Navier-Stokes if weights are not found).
3. Save three output files to `outputs/`:
   - `01_segmentation_mask.png`
   - `02_inpainted_result.png`
   - `03_comparison.png`

To switch to classic inpainting, set `use_ai = False` in `main.py`.

### Option 2 – Streamlit Web UI (`app.py`)

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (typically `http://localhost:8501`) in your browser.

**Workflow:**
1. Upload an image using the sidebar.
2. Select an inpainting model (**AI (LaMa)** or **Classic (Navier-Stokes)**).
3. Click **✂️ Open Mask Drawing Window** – an Intelligent Scissors window will open.
4. Draw the mask and close the window (see keyboard shortcuts below).
5. Click **🚀 Run Inpainting** to process the image.

## Intelligent Scissors – Keyboard Shortcuts

| Action | Key / Button |
|---|---|
| Add anchor point | Left mouse click |
| Finish drawing / close contour | Right mouse click or **Enter** |
| Undo last anchor | **Backspace** |
| Delete selected region (inpaint) | **X** |
| Save mask to `mask_result.png` | **S** |
| Exit window | **ESC** |

## Output Files

| File | Description |
|---|---|
| `outputs/01_segmentation_mask.png` | Binary mask (white = region to remove) |
| `outputs/02_inpainted_result.png` | Image with selected region removed |
| `outputs/03_comparison.png` | Side-by-side comparison: Original, Mask, and Result |