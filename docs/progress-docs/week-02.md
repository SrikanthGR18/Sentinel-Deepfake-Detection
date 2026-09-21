# Sentinel – Weekly Progress Diary

## Week 02

**Period:** 31 August 2026 – 6 September 2026

---

# 1. Weekly Objectives

- Integrate the individual Sentinel modules into a reusable end-to-end analysis pipeline.
- Test the interaction between video, temporal, audio, and fusion modules.
- Improve the robustness of the video inference and media-processing workflow.
- Identify weaknesses in the standalone video detection model.

---

# 2. Team Progress

## Srikanth

### Work Done

- Developed and tested the reusable Sentinel analysis pipeline for processing individual media files.
- Integrated the existing **AMMP, video analysis, temporal analysis, SVIM, and DTFE** components.
- Added media validation and metadata extraction for uploaded videos.
- Verified video duration, resolution, FPS, frame count, and available audio/video streams.
- Improved video frame sampling in `models/inference.py`.
- Added safeguards for invalid videos and videos with zero frames.
- Added frame-level REAL/FAKE probability logging to understand individual model predictions.
- Tested the pipeline using real, FaceForensics++ deepfake, and externally generated AI videos.
- Investigated the incorrect predictions produced by the standalone EfficientNet-B0 model.

### Challenges Faced

- The standalone video model produced inconsistent frame-level predictions for some test videos.
- Some real videos were incorrectly classified as fake.
- The Gemini-generated fake video continued to receive extremely high REAL probabilities.
- Initial frame aggregation did not provide sufficient evidence for reliable detection.

### Solutions

- Improved frame sampling using uniformly distributed frame indices.
- Added frame-level probability diagnostics to identify how individual frames influence the final prediction.
- Compared frame-level predictions with the final averaged video probability.
- Tested an experimental suspicious-frame aggregation approach to investigate whether individual anomalous frames could improve detection.
- Removed the experimental aggregation after observing false positives and continued with the original probability aggregation.

### Results

- The video inference pipeline became more robust to invalid and short videos.
- Frame-level diagnostics provided clearer insight into model behaviour.
- Confirmed that the Gemini failure was not caused by incorrect class-label mapping or a simple frame-aggregation issue.
- Established that further investigation should focus on the training data, model training, and generalization.

---

## Teammate 2 – [Name]

### Work Done

-

### Challenges Faced

-

### Solutions

-

### Results

-

---

## Teammate 3 – [Name]

### Work Done

-

### Challenges Faced

-

### Solutions

-

### Results

-

---

## Teammate 4 – [Name]

### Work Done

-

### Challenges Faced

-

### Solutions

-

### Results

-

---

# 3. Team-Level Progress

-

---

# 4. Testing / Results

-

---

# 5. Problems Encountered

-

---

# 6. Current Project Status

### Completed

- [x] Reusable Sentinel analysis pipeline
- [x] Media validation and metadata extraction
- [x] Improved video frame sampling
- [x] Frame-level model diagnostics
- [x] Video inference testing

### In Progress

- [ ] Improving video-model generalization
- [ ] Training and evaluating the updated EfficientNet-B0 model
- [ ] Expanded testing of the complete Sentinel pipeline

### Pending

- [ ] Browser extension integration
- [ ] Final system evaluation

---

# 7. Plan for Next Week

- Inspect the video-model training and dataset pipeline.
- Verify the REAL/FAKE class mapping and training configuration.
- Retrain the EfficientNet-B0 model using the prepared Celeb-DF-v2 dataset.
- Evaluate training and validation performance.
- Test the newly trained model on the external evaluation videos.

---

# 8. Guide Remarks

**Guide:**  

**Date:**  

**Remarks:**  

---