# FOG_ASSIGNMENT
Studio-Quality Portrait Enhancement Pipeline
Developed by Yuvraj Jain | ML Engineer Candidate
This repository contains a modular, intelligent computer vision pipeline designed to transform raw, uncontrolled human portraits into studio-quality images. The system utilizes automated quality assessment to apply targeted enhancements, ensuring natural results with high inference performance.

🚀 Overview
The PortraitEnhancer system addresses common mobile photography artifacts such as motion blur, uneven lighting, cluttered backgrounds, and low contrast. Unlike static filters, this pipeline analyzes the unique characteristics of each input image to decide which processing modules are required.

Core Capabilities:
Intelligent Analysis: Automated detection of blur, contrast, brightness, and sharpness using Laplacian variance and histogram analysis.

Targeted Face Enhancement: Specialized modules for skin smoothing and feature sharpening that preserve natural textures.

Advanced Segmentation: Person-to-background separation using GrabCut combined with guided filtering for soft, natural edges.

Professional Bokeh: Multi-pass Gaussian blurring with distance-transform mapping to simulate high-end optical depth-of-field.

Lighting Correction: LAB color space manipulation to fix underexposure and balance uneven facial shadows without shifting color accuracy.

💻 Technical Stack
Language: Python 3.x

Primary Library: OpenCV (cv2)

Supporting Libraries: NumPy, Typing, Dataclasses

Techniques: GrabCut Segmentation, Guided Filtering, Frequency Separation, CLAHE, Laplacian Variance Detection.

🏃 Getting Started
Prerequisites
Bash

pip install opencv-python numpy
Usage
Place your raw image (e.g., portrait.jpg) in the root directory and run:

Python

from portrait_enhancer import PortraitEnhancer

# Initialize with debug=True to see real-time image analysis logs
enhancer = PortraitEnhancer(debug=True)

# Process the image
enhancer.enhance_portrait('input_camera.jpg', 'output_studio.jpg')
📊 Sample Output Logs
When running in debug mode, the pipeline provides deep insights into the image quality metrics:

Plaintext

=== IMAGE ANALYSIS ===
Blur Score (Laplacian Variance): 84.12
  → Needs Deblur: True
Contrast Range: 92.00
  → Needs Contrast Enhancement: True
Brightness: 72.45
  → Needs Lighting Correction: True
==============================

✓ Applying: Deblur
✓ Applying: Lighting Correction
✓ Applying: Face Lighting Balance (Shadow Correction)
✓ Applying: Background Blur (Studio Effect)
👤 Contact & Application Details
Candidate: Yuvraj Jain

Contact: workuvj@gmail.com

Role: ML Engineer

Target Company: FOG Technologies

This project was developed to demonstrate expertise in computer vision, modular software design, and automated image quality assessment for professional-grade photography applications.
