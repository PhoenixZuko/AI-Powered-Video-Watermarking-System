import os
import cv2
import numpy as np
import yaml
import onnxruntime as ort
from watermark.jnd_watermark import JndWatermark
from watermark.dwt_watermark import DwtWatermark
from watermark.onnx_inference import inference

# 🔹 Încarcă setările din YAML
with open("profiles/watermark_settings.yaml", "r") as f:
    config = yaml.safe_load(f)

# 🔹 Activăm sau dezactivăm watermarking-ul în funcție de YAML
APPLY_JND = config["watermark"]["apply_jnd"]
APPLY_DWT = config["watermark"]["apply_dwt"]
APPLY_ONNX = config["watermark"]["apply_onnx"]
USE_GPU = config["watermark"].get("use_gpu", False)  # Opțiune pentru GPU

# ✅ Inițializăm JND și DWT Watermark
jnd_processor = JndWatermark(region_size=55, strength=50)
dwt_processor = DwtWatermark(password_wm=3, wm_size=64)

# 🔹 Selectează provider-ul pentru ONNX (CPU/GPU)
providers = ["CPUExecutionProvider"]  # Implicit folosim CPU
if USE_GPU:
    try:
        ort.InferenceSession("watermark/weights/deeplabv3.onnx", providers=["CUDAExecutionProvider"])
        providers = ["CUDAExecutionProvider"]
        print("✅ ONNX va folosi GPU (CDU)")
    except Exception as e:
        print(f"⚠️ GPU indisponibil, folosim CPU. Eroare: {e}")

# ✅ Încarcă modelul ONNX doar dacă este activat în YAML
onnx_session = None
if APPLY_ONNX:
    onnx_session = ort.InferenceSession("watermark/weights/deeplabv3.onnx", providers=providers)

# 🔹 Funcție pentru procesarea videoclipului
def process_video(input_path, output_path):
    print(f"🎥 Processing video: {input_path}")

    cap = cv2.VideoCapture(input_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        print(f"📌 Processing frame {frame_count}...", end="\r")

        pseudo_id = frame_count  # 🔹 Creăm un ID unic pentru fiecare cadru

        # ✅ Aplicăm detecția ONNX pentru față/păr (CPU/GPU automat)
        if APPLY_ONNX and onnx_session:
            hair_mask_resized, hair_mask, face_pixels = inference(frame, onnx_session)
            if hair_mask is not None and np.any(hair_mask):
                frame = cv2.addWeighted(frame, 0.8, hair_mask_resized, 0.2, 0)  

        # ✅ Aplicăm JND Watermark
        if APPLY_JND:
            dense_regions = jnd_processor.init_regions(frame)  # ✅ Inițializăm zonele dense

        # ✅ Aplicăm DWT Watermark
        if APPLY_DWT:
            frame = dwt_processor.embed_dwt_watermark(frame, pseudo_id)
            frame = jnd_processor.embed_pseudo_random_id_color(frame, pseudo_id, dense_regions)  

        out.write(frame)

    cap.release()
    out.release()
    print(f"\n✅ Video processing complete! Saved as: {output_path}")

if __name__ == "__main__":
    process_video("input.mp4", "output_watermarked.mp4")
