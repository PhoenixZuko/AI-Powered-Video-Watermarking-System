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

