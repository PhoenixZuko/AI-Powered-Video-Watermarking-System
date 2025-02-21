"""
📌 ONNX-Based Face & Hair Detection for Video Watermarking
---------------------------------------------------------
🔹 Author: Andrei Sorin Ștefan & ChatGPT
🔹 Description:
    This script loads an ONNX model and performs real-time segmentation of **hair and face regions**.
    It applies selective pixel modifications to prevent clean video duplication.

🔹 Features:
    - Prepares images for ONNX inference by resizing and normalizing.
    - Detects **face and hair pixels** and applies small, randomized modifications.
    - Returns an interference mask that slightly alters detected regions.

🔹 Dependencies:
    - OpenCV, NumPy, ONNXRuntime, TorchVision

🔹 Usage:
    - Load an ONNX session and pass an image frame to `inference(image, onnx_session)`.
"""

import onnxruntime as ort
import numpy as np
import torchvision.transforms as transforms
import cv2

# 🔹 Prepare image for ONNX model
def prepare_image(image):
    """
    Converts an image into an ONNX-compatible format:
    - Converts to tensor
    - Normalizes values based on model requirements
    - Expands dimensions for batch processing
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    image_tensor = transform(image)
    image_batch = image_tensor.unsqueeze(0)
    return image_batch.numpy()

# 🔹 Inference function for ONNX-based segmentation
def inference(image, onnx_session):
    """
    Runs ONNX inference on an image to detect **hair and face** pixels.
    Randomly applies small pixel modifications to interfere with screen recording attempts.
    
    Args:
        image (numpy.ndarray): Input image frame.
        onnx_session (onnxruntime.InferenceSession): Loaded ONNX session.

    Returns:
        final_mask_resized (numpy.ndarray): Mask resized to original frame dimensions.
        final_mask (numpy.ndarray): Interference mask for visualization.
        face_pixels (tuple): Coordinates of detected face pixels.
    """
    if image is None:
        return None, None, None

    height, width = image.shape[:2]

    # Resize image to match ONNX model input size (224x224)
    resized_image = cv2.resize(image, (224, 224))
    
    # Convert image to ONNX-compatible format
    transformed_image = prepare_image(resized_image)
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name

    # Run inference
    output = onnx_session.run([output_name], {input_name: transformed_image})[0]
    predicted_mask = output.squeeze(0).argmax(0)

    # 🔹 Create an empty mask with 3 color channels
    final_mask = np.zeros((predicted_mask.shape[0], predicted_mask.shape[1], 3), dtype=np.uint8)

    # Detect pixels corresponding to hair and face
    hair_pixels = np.where(predicted_mask == 15)
    face_pixels = np.where((predicted_mask == 1) | (predicted_mask == 2))

    # 🔹 Apply small interference by modifying only a few pixels
    if hair_pixels[0].size > 0:
        for i in range(len(hair_pixels[0])):
            if np.random.rand() < 0.02:  # ❗ Modify only 2% of hair pixels
                final_mask[hair_pixels[0][i], hair_pixels[1][i], :] = [0, 0, 0]  # Black for hair interference

    if face_pixels[0].size > 0:
        for i in range(len(face_pixels[0])):
            if np.random.rand() < 0.02:  # ❗ Modify only 2% of face pixels
                final_mask[face_pixels[0][i], face_pixels[1][i], :] = [255, 255, 255]  # White for face interference

    # Resize the interference mask back to the original frame size
    final_mask_resized = cv2.resize(final_mask, (width, height))

    return final_mask_resized, final_mask, face_pixels
