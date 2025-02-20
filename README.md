AI-Powered Video Watermarking System (In Development)

  Overview

This project enhances traditional image watermarking techniques by adapting them for video processing. It applies multiple layers of watermarking to improve security and prevent unauthorized duplication, including screen recording attacks.

  Note: This project is currently private and under active development. The repository will be updated as progress continues.

  How It Works – Modular Breakdown

  1. Hidden Watermarks (DWT & JND)

DWT (Discrete Wavelet Transform) – Embeds subtle watermarks resistant to compression and filtering.

JND (Just Noticeable Difference) – Ensures watermark visibility remains imperceptible to the human eye while still detectable upon analysis.

  2. AI-Assisted Dynamic Marking (ONNX Inference)

Analyzes each frame in real-time, detecting faces and hair as primary watermarking zones.

Applies 1-2 enlarged pixels per frame, particularly in the facial region and detected hair areas, ensuring that the main subject of the video remains highly protected.

Enhances protection against screen recording by introducing dynamic pixel variations, making unauthorized duplication more difficult.

  3. Visible Filigree Watermarks

Includes aesthetic watermark overlays that remain visible during playback.

Can be revealed or reinforced through forensic analysis, ensuring additional security layers.

 Technologies Used

This project is built using:

 Python – Core scripting language
 OpenCV – Image and video processing
 ONNX Runtime – AI-based face and hair detection
 NumPy – Efficient matrix and numerical operations
 Torchvision – Image transformations and processing
 DWT (Discrete Wavelet Transform) – Invisible watermark embedding
 JND (Just Noticeable Difference) – Adaptive watermarking
 Custom Zodiac AI (In Development) – Future-proofing content protection

 Future Enhancements – The Zodiac AI

A key future goal is to integrate Zodiac AI, a system designed to:

 Test and validate applied watermarks to ensure every frame is properly protected.
  Fingerprint content ownership by associating watermarks with unique author metadata.
  Train an advanced AI model capable of verifying rightful ownership beyond standard watermark detection.

 I am looking to collaborate with AI training specialists to enhance Zodiac's capabilities, aiming for an industry-standard tool for digital content protection.

