# Sentinel – Weekly Progress Diary

## Week 04

**Period:** 14 September 2026 – 20 September 2026

---

# 1. Weekly Objectives

- Evaluate the newly trained EfficientNet-B0 model on unseen test videos.
- Compare the updated model against the previous baseline.
- Analyze model errors and generalization across different sources.
- Document the current video-model and Sentinel pipeline results.

---

# 2. Team Progress

## Srikanth

### Work Done

- Executed the updated video-model evaluation using the newly trained `best_model.pth`.
- Evaluated the model on 11 test videos:
  - 5 FaceForensics++ real videos
  - 5 FaceForensics++ deepfake videos
  - 1 externally generated Gemini AI video
- Recorded frame-level REAL/FAKE probabilities for each test video.
- Calculated video-level predictions using averaged frame probabilities.
- Calculated Accuracy, Precision, Recall, F1 Score, and confusion-matrix values.
- Compared the updated model performance with the previous baseline.
- Investigated individual false-positive and false-negative cases.
- Evaluated the generalization of the trained model beyond the Celeb-DF-v2 training distribution.
- Verified that the Gemini video continued to be classified as authentic by the standalone video model.
- Reviewed the role of multimodal analysis in handling cases where the standalone video model produced incorrect predictions.

### Challenges Faced

- The updated model achieved high validation accuracy but lower performance on the external video test set.
- The model incorrectly classified some FaceForensics++ fake videos as authentic.
- The externally generated Gemini video was still classified as authentic with very high REAL probability.
- The difference between validation performance and external test performance indicated limited cross-dataset/generalization capability.

### Solutions

- Performed detailed video-level and frame-level evaluation rather than relying only on overall accuracy.
- Compared individual predictions to identify false positives and false negatives.
- Retained the original probability aggregation instead of using the experimental suspicious-frame aggregation because it produced false positives.
- Used the existing temporal and audio modules as complementary evidence within the Sentinel multimodal pipeline.
- Documented the current limitations for further model and pipeline improvement.

### Results

- Updated standalone video-model evaluation:
  - Accuracy: **63.64%**
  - Precision: **75.00%**
  - Recall: **50.00%**
  - F1 Score: **60.00%**

- Confusion matrix:
  - True Positive: **3**
  - True Negative: **4**
  - False Positive: **1**
  - False Negative: **3**

- FaceForensics++ real videos: **4/5 correctly classified**
- FaceForensics++ fake videos: **3/5 correctly classified**
- Gemini AI-generated video: **incorrectly classified as Authentic**

- Compared with the previous baseline, performance improved from:
  - Accuracy: **54.55% → 63.64%**
  - Precision: **66.67% → 75.00%**
  - Recall: **33.33% → 50.00%**
  - F1 Score: **44.44% → 60.00%**

- The trained model achieved **95.04% validation accuracy on Celeb-DF-v2** but showed lower performance on the external test set, highlighting the need for improved cross-dataset generalization.

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

- [x] EfficientNet-B0 model training
- [x] Validation evaluation
- [x] External video evaluation
- [x] Updated performance metrics
- [x] Error and failure-case analysis
- [x] Comparison with previous baseline

### In Progress

- [ ] Improving cross-dataset and cross-generator generalization
- [ ] Expanded multimodal evaluation
- [ ] Further validation of Sentinel trust-score fusion

### Pending

- [ ] Browser extension integration
- [ ] Large-scale final evaluation
- [ ] Final system validation

---

# 7. Plan for Next Week

- Continue evaluation of the complete multimodal Sentinel pipeline.
- Compare standalone video predictions with multimodal DTFE results.
- Expand testing with additional real, manipulated, and AI-generated media.
- Investigate improvements to model generalization and trust-score fusion.
- Continue integration and validation of the remaining Sentinel components.

---

# 8. Guide Remarks

**Guide:**  

**Date:**  

**Remarks:**  