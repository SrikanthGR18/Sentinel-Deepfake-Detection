# SENTINEL PROJECT SPECIFICATION

Project Name:
Sentinel: A Deepfake Detection and Digital Trust Layer for Online Media

Project Goal:
Develop a practical multimodal deepfake detection system that analyzes both video and audio content and generates an interpretable trust score.

Current Objective:
Refactor and improve the existing prototype so that it follows the architecture defined in the project report and research paper.

Required Architecture:

Media Upload
↓
Adaptive Multi-Modal Preprocessing (AMMP)
↓
Video Analysis
↓
Audio Analysis
↓
Temporal Consistency Analysis
↓
Trust Score Fusion
↓
Dashboard Visualization

---

AMMP MODULE

Purpose:
Dynamically preprocess media based on quality.

Video Quality Parameters:

* Blur Detection
* Brightness Analysis
* Contrast Analysis
* Motion Consistency

Audio Quality Parameters:

* Noise Detection
* Signal Variance
* Audio Energy

Actions:

* Frame Selection
* Frame Enhancement
* Audio Normalization
* Audio Denoising

Output:
Preprocessed Video Frames
Preprocessed Audio

---

VIDEO ANALYSIS MODULE

Purpose:
Analyze visual artifacts.

Features:

* Face Detection
* Texture Inconsistency Detection
* Compression Artifact Detection
* Facial Blending Detection

Preferred Models:

* EfficientNet-B0
  or
* XceptionNet

Output:
Video Confidence Score (0-100)

---

TEMPORAL ANALYSIS MODULE

Purpose:
Analyze frame-to-frame consistency.

Features:

* Optical Flow Analysis
* Motion Consistency
* Sudden Frame Changes
* Lip Movement Consistency

Preferred Tool:
OpenCV Optical Flow

Output:
Temporal Confidence Score (0-100)

---

AUDIO ANALYSIS MODULE

Purpose:
Detect synthetic speech characteristics.

Features:

* MFCC
* Spectral Contrast
* RMS Energy
* Zero Crossing Rate

Preferred Libraries:

* Librosa
* NumPy

Output:
Audio Confidence Score (0-100)

---

TRUST SCORE FUSION MODULE

Video Score:

VideoScore =
0.6 × SpatialScore
+
0.4 × TemporalScore

Audio Score:

AudioScore =
0.7 × MFCCScore
+
0.3 × AcousticPatternScore

Final Score:

FinalScore =
0.6 × VideoScore
+
0.4 × AudioScore

Classification:

0-40 → Fake

41-70 → Suspicious

71-100 → Authentic

Output:
Final Trust Score
Final Classification

---

DASHBOARD REQUIREMENTS

Display:

* Uploaded Media Preview
* Video Confidence Score
* Audio Confidence Score
* Temporal Score
* Final Trust Score
* Classification Result

Visualization:

* Progress Bars
* Score Cards
* Charts

---

FUTURE FEATURES

* Browser Extension
* Real-Time Webcam Detection
* Cloud Deployment
* Explainable AI Heatmaps

These are future features and should not be implemented before the core architecture is completed.

Current Priority:
Implement the architecture exactly as specified above.
