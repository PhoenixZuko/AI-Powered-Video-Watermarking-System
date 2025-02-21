# AI-Powered Video Watermarking System (In Development)

## Overview
This project enhances traditional image watermarking techniques by adapting them for **video processing**. It applies **multi-layered watermarking** to improve security and prevent unauthorized duplication, including **screen recording attacks**.

### **Project Goals**
We aim to develop a **robust AI-driven watermarking system** that prevents video piracy by integrating:
- **Invisible Watermarking (DWT & JND)** – Ensures imperceptible but detectable marks.
- **AI-Assisted Dynamic Marking (ONNX Inference)** – Applies interference to prevent screen recording.
- **Visible Filigree Watermarks** – Strengthens ownership attribution.
- **Future Integration with Zodiac AI** – Advanced content fingerprinting and forensic validation.

### **Potential Use Cases**
- **DRM (Digital Rights Management)** – Protects premium video content.
- **Forensic Video Analysis** – Identifies modified or pirated footage.
- **Content Ownership Protection** – Ensures rightful attribution.

---

## **How It Works – Modular Breakdown**

### **1️ Hidden Watermarks (DWT & JND)**
 **DWT (Discrete Wavelet Transform)** – Embeds subtle watermarks resistant to compression and filtering.  
 **JND (Just Noticeable Difference)** – Ensures watermark visibility remains imperceptible to the human eye but detectable upon analysis.  

### **2️ AI-Assisted Dynamic Marking (ONNX Inference)**
 Analyzes each frame in real-time, detecting **faces and hair** as primary watermarking zones.  
 **Pixel-based interference**: Adds 1-2 enlarged pixels per frame in facial and hair areas, preventing clean duplication.  
 **Prevents screen recording leaks** through dynamic pixel variation.  

### **3️ Visible Filigree Watermarks**
 Adds aesthetic watermark overlays during playback.  
 Can be **strengthened through forensic analysis** to detect modifications.  

---

## **Technologies Used**
This project is built using:
- **Python** – Core scripting language  
- **OpenCV** – Image & video processing  
- **ONNX Runtime** – AI-powered inference for face/hair detection  
- **NumPy** – Efficient numerical operations  
- **Torchvision** – Image transformations  
- **DWT (Discrete Wavelet Transform)** – Invisible watermark embedding  
- **JND (Just Noticeable Difference)** – Adaptive watermarking  
- **Custom Zodiac AI (In Development)** – Future-proofing content protection  

---

# AI-Powered Video Watermarking System (ONNX + JND + DWT)

## Overview
This project applies multiple layers of **forensic video watermarking** using **AI inference, Just Noticeable Difference (JND), and Discrete Wavelet Transform (DWT)** to enhance security and prevent unauthorized duplication.

---

##  System Requirements

###  Minimum Requirements (CPU-Based)
- **OS:** Windows 10+, Ubuntu 20.04+, macOS (Limited Support)
- **Python Version:** Python **3.8+**
- **RAM:** At least **8GB**
- **Storage:** Minimum **5GB free space**

###  Recommended for GPU (CUDA) Acceleration
- **GPU:** NVIDIA **RTX 2060 or better** (minimum **4GB VRAM**)
- **CUDA Version:** **CUDA 11.0+**
- **Drivers:** Ensure **NVIDIA Drivers** are up to date

---

##  1. Installation Guide

### **1️ Clone the Repository**
```bash
git clone https://github.com/PhoenixZuko/AI-Video-Watermark.git
cd AI-Video-Watermark
```

### **2️ Create & Activate Virtual Environment (Recommended)**
```bash
python3 -m venv venv  # Create virtual environment
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows
```

### **3️ Install Dependencies**
```bash
pip install -r requirements.txt
```

---

##  2. Install & Configure ONNX (CPU/GPU)

### **1️ Check If GPU is Available**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
If it returns `True`, your GPU supports CUDA.

### **2️ Install ONNX & CUDA Dependencies**
####  For CPU Only:
```bash
pip install onnxruntime
```
####  For GPU (CUDA) Acceleration:
```bash
pip install onnxruntime-gpu
```

### **3️ Verify ONNX Installation**
```python
import onnxruntime as ort
print("Available ONNX Execution Providers:", ort.get_available_providers())
```
If it shows **CUDAExecutionProvider**, ONNX is using your GPU.

---

##  3. Download & Configure ONNX Model

### **1️ Create Required Folders**
```bash
mkdir -p watermark/weights profiles results
```

### **2️ Download ONNX Model (DeepLabV3)**
```bash
wget -O watermark/weights/deeplabv3.onnx https://github.com/onnx/models/raw/main/vision/segmentation/deeplabv3/deeplabv3.onnx
```
For Windows users, **download manually** from:
 [DeepLabV3 ONNX Model](https://github.com/onnx/models/raw/main/vision/segmentation/deeplabv3/deeplabv3.onnx)
Then move it to **watermark/weights/**.

---

##  4. Configure YAML Settings
Edit the `profiles/watermark_settings.yaml` file to customize settings:
```yaml
watermark:
  apply_jnd: true
  apply_dwt: true
  apply_onnx: true
  use_gpu: true   # Change to false if using CPU
```

---

##  5. Run the Program

### **1️ Test ONNX Detection**
```bash
python onnx_test.py
```
Expected output:
```
 ONNX Model Loaded Successfully!
 ONNX is running on CUDA (GPU) / CPU
```

### **2️ Process Video with Watermarking**
####  Run CPU Version:
```bash
python main.py
```
####  Run GPU (CUDA) Version:
```bash
python main_CDU.py
```

---

##  6. Output & Results
Processed videos are saved in the `results/` folder:
```bash
results/output_watermarked.mp4
```

---

##  7. Troubleshooting

### ** ONNX Fails to Run?**
Try **forcing CPU execution** by modifying `profiles/watermark_settings.yaml`:
```yaml
use_gpu: false
```

### ** CUDA Not Detected?**
Make sure your system has:
```bash
nvidia-smi  # Checks if NVIDIA GPU is detected
nvcc --version  # Checks if CUDA is installed
```

### ** Video Processing Too Slow?**
- **Enable GPU (`use_gpu: true`)**
- Use a **lower resolution** input video
- Reduce **watermark strength** in YAML

---

##  8. Summary

| Step | Command |
|------|---------|
| **Clone Project** | `git clone https://github.com/PhoenixZuko/AI-Video-Watermark.git` |
| **Create Virtual Env** | `python3 -m venv venv && source venv/bin/activate` |
| **Install Dependencies** | `pip install -r requirements.txt` |
| **Install ONNX (CPU)** | `pip install onnxruntime` |
| **Install ONNX (GPU)** | `pip install onnxruntime-gpu` |
| **Download ONNX Model** | `wget -O watermark/weights/deeplabv3.onnx <URL>` |
| **Run CPU Version** | `python main.py` |
| **Run GPU Version** | `python main_CDU.py` |
| **Check CUDA Support** | `python -c "import torch; print(torch.cuda.is_available())"` |

---

 **Now you are ready to use AI-powered video watermarking!** 



## **Future Enhancements – The Zodiac AI**
 A key future goal is to integrate **Zodiac AI**, a system designed to:
 **Validate applied watermarks** to ensure every frame is properly protected.  
 **Fingerprint content ownership** by associating watermarks with unique metadata.  
 **Train an advanced AI model** capable of verifying rightful ownership beyond standard detection.  

 *Looking to collaborate with AI training specialists to enhance Zodiac’s capabilities and develop an industry-standard tool for digital content protection.*  

---

## **Recent Updates**
 Successfully implemented **CDU** for improved processing speed.  
 Added **multiple watermarking profiles** for different levels of protection.  
 Optimized **ONNX inference** for balanced face detection.  
 Currently testing **various filming conditions** (close-up, wide shots).  

---

## **Contributing & Contact**
 **Contributions are welcome!** If you're interested in improving AI watermarking or forensic analysis, feel free to open an issue or submit a pull request.  

 **Contact:**  
- **Email:** andrei.sorin.stefan@gmail.com  
- **GitHub:** [PhoenixZuko](https://github.com/PhoenixZuko)  

---

## **License**
📜 This project is released under the MIT License.  

