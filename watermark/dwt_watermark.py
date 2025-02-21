"""
📌 DWT-Based Watermarking System for Video Protection
-----------------------------------------------------
🔹 Author: Andrei Sorin Ștefan & ChatGPT
🔹 Description:
    This module implements **Discrete Wavelet Transform (DWT) watermarking** for images and videos,
    ensuring robust protection against unauthorized duplication.

    Features:
    - **DWT-based embedding** with a pseudo-random sequence for security.
    - **Contrast enhancement** to reinforce watermark visibility.
    - **Watermark decryption & extraction** using a shuffling mechanism.

🔹 Dependencies:
    - OpenCV, NumPy, WaterMarkCore (custom DWT processor)

🔹 Usage:
    - Import `DwtWatermark` and use `.embed_dwt_watermark()` to apply a watermark.
"""

import cv2
import numpy as np
from .dwt_core import WaterMarkCore


class DwtWatermark:
    """Implements DWT-based watermarking for image and video processing."""

    def __init__(self, password_wm=3, wm_size=64):
        """
        Initializes the watermarking system.

        Args:
            password_wm (int): Seed for shuffling the watermark sequence.
            wm_size (int): Size of the watermark in bits.
        """
        self.password_wm = password_wm
        self.wm_size = wm_size

    def extract_decrypt(self, wm_avg):
        """
        Extracts and enhances the watermark after decryption.

        Args:
            wm_avg (numpy.ndarray): Extracted watermark in raw format.

        Returns:
            numpy.ndarray: Enhanced watermark with increased contrast.
        """
        wm_index = np.arange(self.wm_size)
        np.random.RandomState(self.password_wm).shuffle(wm_index)
        wm_avg[wm_index] = wm_avg.copy()

        # 🔹 Enhance watermark contrast for better visibility
        wm_avg = np.clip(wm_avg * 1.5, 0, 1)
        return wm_avg

    def embed_dwt_watermark(self, image, pseudo_id):
        """
        Embeds a **DWT watermark** into an image frame.

        Args:
            image (numpy.ndarray): Input image frame to watermark.
            pseudo_id (int): Unique ID to embed as a watermark.

        Returns:
            numpy.ndarray: Watermarked image.
        """
        bwm_c = WaterMarkCore(password_img=3)
        bwm_c.read_img_arr(img=image)

        # Convert pseudo_id to a 64-bit binary sequence
        bit_sequence = [int(bit) for bit in f"{pseudo_id:064b}"]
        wm_bit = np.array(bit_sequence)
        np.random.RandomState(self.password_wm).shuffle(wm_bit)

        # Apply the shuffled watermark sequence
        bwm_c.read_wm(wm_bit)

        # 🔹 Embed watermark into the image
        embed_img = bwm_c.embed()

        # 🔹 Increase image contrast to reinforce watermark visibility
        embed_img = np.clip(embed_img * 1.2, 0, 255).astype(np.uint8)

        return embed_img

    def decode_dwt_watermark(self, watermarked_image):
        """
        Extracts and decrypts a DWT watermark from a watermarked image.

        Args:
            watermarked_image (numpy.ndarray): Watermarked image for extraction.

        Returns:
            int: Extracted watermark ID.
        """
        print("🔎 Decoding DWT watermark...")
        bwm_c = WaterMarkCore(password_img=3)

        # Extract raw watermark using KMeans clustering
        wm_avg = bwm_c.extract_with_kmeans(img=watermarked_image, wm_shape=self.wm_size)

        # Decrypt and enhance the watermark
        wm = self.extract_decrypt(wm_avg=wm_avg)

        # Convert watermark bits to integer ID
        byte = ''.join(str((i >= 0.5) * 1) for i in wm)
        wm = int("".join(map(str, byte)), 2)

        return wm
