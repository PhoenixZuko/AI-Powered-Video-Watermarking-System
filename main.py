"""
📌 AI-Powered Video Watermarking System
--------------------------------------
🔹 Author: Andrei Sorin Ștefan & ChatGPT
🔹 Description:
    This script applies multiple layers of watermarking to video files to enhance security and prevent unauthorized duplication. 
    It includes:
      - JND Watermark (Just Noticeable Difference)
      - DWT Watermark (Discrete Wavelet Transform)
      - ONNX AI-based interference (face & hair detection)

🔹 Features:
    - Processes video frame-by-frame and applies selected watermarking techniques.
    - Uses an ONNX deep learning model to detect faces & hair, ensuring watermark placement in key regions.
    - Configurable settings via YAML file.

🔹 Dependencies:
    - OpenCV, NumPy, PyYAML, ONNXRuntime

🔹 Usage:
    - Modify `watermark_settings.yaml` to enable/disable specific watermarking methods.
    - Run the script with an input video file.

🔹 Example:
    python process_video.py input.mp4 output_watermarked.mp4
"""

import os
import cv2
import numpy as np
import yaml
import onnxruntime as ort
from watermark.jnd_watermark import JndWatermark
from watermark.dwt_watermark import DwtWatermark
from watermark.onnx_inference import inference


# 🔹 Load settings from YAML configuration
with open("profiles/watermark_settings.yaml", "r") as f:
    config = yaml.safe_load(f)

# 🔹 Enable/disable watermarking techniques based on YAML settings
APPLY_JND = config["watermark"]["apply_jnd"]
APPLY_DWT = config["watermark"]["apply_dwt"]
APPLY_ONNX = config["watermark"]["apply_onnx"]

# ✅ Initialize watermarking modules
jnd_processor = JndWatermark(region_size=55, strength=50)
dwt_processor = DwtWatermark(password_wm=3, wm_size=64)

# 🔹 Video processing function
def process_video(input_path, output_path):
    print(f"🎥 Processing video: {input_path}")

    cap = cv2.VideoCapture(input_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

    # ✅ Load ONNX model only if enabled in settings
    if APPLY_ONNX:
        onnx_session = ort.InferenceSession("./watermark/weights/deeplabv3.onnx")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        print(f"📌 Processing frame {frame_count}...", end="\r")

        pseudo_id = frame_count  # 🔹 Generate a unique ID for each frame

        # ✅ Apply ONNX-based face & hair detection
        if APPLY_ONNX:
            hair_mask_resized, hair_mask, face_pixels = inference(frame, onnx_session)
            if hair_mask is not None and np.any(hair_mask):
                frame = cv2.addWeighted(frame, 0.8, hair_mask_resized, 0.2, 0)  

        # ✅ Apply JND Watermark
        if APPLY_JND:
            dense_regions = jnd_processor.init_regions(frame)  # ✅ Initialize dense regions

        # ✅ Apply DWT Watermark
        if APPLY_DWT:
            frame = dwt_processor.embed_dwt_watermark(frame, pseudo_id)
            frame = jnd_processor.embed_pseudo_random_id_color(frame, pseudo_id, dense_regions)  

        out.write(frame)

    cap.release()
    out.release()
    print(f"\n✅ Video processing complete! Saved as: {output_path}")

if __name__ == "__main__":
    process_video("input.mp4", "output_watermarked.mp4")
