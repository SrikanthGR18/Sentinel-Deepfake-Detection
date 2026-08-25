# Sentinel – Weekly Progress Diary

## Week 01

**Period:** 24 August 2026 – 30 August 2026

---

# 1. Weekly Objectives

- Establish a baseline for the existing video-based deepfake detection model.
- Evaluate the model using real and manipulated videos from the FaceForensics++ dataset and an externally generated AI video.
- Integrate and test the Sentinel multimodal analysis pipeline consisting of AMMP, video analysis, temporal analysis, SVIM audio analysis, and DTFE trust-score fusion.

---

# 2. Team Progress

## Srikanth

### Work Done

- Prepared a baseline evaluation setup for Sentinel using FaceForensics++ test videos.
- Added 5 real and 5 deepfake videos from FaceForensics++ to the test dataset.
- Added an externally generated Gemini AI video as an additional test case.
- Implemented and executed the baseline video evaluation across 11 test videos.
- Recorded video-level predictions, real/fake probabilities, votes, scores, and evaluation metrics.
- Established the baseline performance of the existing video-based detection model.
- Created a reusable Sentinel analysis pipeline connecting the existing Sentinel components:
  - AMMP preprocessing
  - Video/spatial analysis
  - Temporal analysis
  - SVIM audio analysis
  - DTFE trust-score fusion
- Created and executed a pipeline test script for testing the complete Sentinel analysis flow.
- Verified FFmpeg availability and investigated audio streams in the evaluation videos.
- Tested the complete Sentinel pipeline using the externally generated Gemini AI video.
- Documented the baseline and multimodal experiment results.
- Synchronized the development branch with the latest changes from the main branch.
  
### Challenges Faced

- The initial video testing script expected three return values while the prediction function returned a dictionary, causing a value-unpacking error.
- Incorrect video paths initially caused OpenCV to fail when opening test videos.
- The standalone video model produced incorrect predictions for several FaceForensics++ deepfake samples.
- The video model classified the externally generated Gemini AI video as authentic with 100% real probability.
- The FaceForensics++ videos used for testing did not contain audio streams, preventing SVIM from being evaluated on those samples.
- The multimodal pipeline required careful integration of outputs from AMMP, temporal analysis, SVIM, and DTFE.
- SVIM module files were updated and renamed during team integration, requiring the local branch to be synchronized with the latest main branch.

### Solutions

- Updated the testing code to consume the prediction dictionary returned by the video inference module.
- Verified file paths and used direct OpenCV tests to confirm that videos could be opened and frames extracted.
- Used a dedicated baseline evaluation script to test multiple real and fake videos consistently.
- Added an external AI-generated video to evaluate generalization beyond the training/test dataset.
- Used FFmpeg to inspect video streams and confirm the absence or presence of audio.
- Reused the existing AMMP audio preprocessing and extraction functions rather than duplicating audio-processing logic.
- Built a reusable analysis pipeline that passes the outputs of each Sentinel component into the next stage and finally into DTFE fusion.
- Merged the latest `main` branch changes into the local development branch to reflect the updated SVIM module structure.

### Results

- Baseline evaluation was completed on 11 videos:
  - 5 real FaceForensics++ videos
  - 5 fake FaceForensics++ videos
  - 1 externally generated Gemini AI video
- Baseline performance:
  - Accuracy: 54.55%
  - Precision: 66.67%
  - Recall: 33.33%
  - F1 Score: 44.44%
- The baseline model correctly classified 6 out of 11 test videos.
- The complete multimodal pipeline was successfully executed on the Gemini AI-generated video.
- Gemini video pipeline result:
  - Video score: 100.00
  - Temporal score: 52.00
  - MFCC score: 49.00
  - Acoustic pattern score: 39.00
  - Audio score: 46.00
  - Final DTFE score: 66.90
  - Classification: Suspicious
  - Confidence: Medium
- The multimodal pipeline therefore detected suspicious characteristics in the Gemini video despite the standalone video model classifying it as authentic.

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

- Established a working baseline for the existing video-based deepfake detection model.
- Prepared a controlled evaluation set containing real, FaceForensics++ manipulated, and externally generated AI content.
- Integrated the major Sentinel analysis components into a reusable multimodal pipeline.
- Successfully connected AMMP preprocessing, video analysis, temporal analysis, SVIM audio analysis, and DTFE trust-score fusion.
- Demonstrated that multimodal analysis can produce a different and more cautious assessment than the standalone video model on an externally generated AI video.
- Continued integration of the SVIM module with the main project branch.

---

# 4. Testing / Results

| Test | Result |
|---|---|
| FaceForensics++ real videos | 4/5 correctly classified |
| FaceForensics++ fake videos | 2/5 correctly classified |
| External Gemini AI video | Classified as Authentic by video model |
| Baseline accuracy | 54.55% |
| Baseline precision | 66.67% |
| Baseline recall | 33.33% |
| Baseline F1 score | 44.44% |
| Full pipeline on Gemini video | Suspicious |
| Gemini final DTFE score | 66.90 |
| Gemini pipeline confidence | Medium |

---

# 5. Problems Encountered

| Problem | Person Responsible | Solution | Status |
|---|---|---|---|
| Prediction function returned a dictionary instead of three values | Srikanth | Updated the test script to use dictionary fields | Resolved |
| Incorrect test video path | Srikanth | Verified project-relative paths and file existence | Resolved |
| Weak baseline performance on FaceForensics++ deepfakes | Srikanth | Established baseline and continued investigation through multimodal analysis | In Progress |
| Gemini AI video classified as real by video model | Srikanth | Added temporal and audio analysis followed by DTFE fusion | Resolved at pipeline level |
| FaceForensics++ test videos had no audio streams | Srikanth | Verified streams using FFmpeg; used external AI video for audio-enabled pipeline testing | Resolved |
| SVIM module file structure changed during integration | Shivam | Synchronized local branch with latest `main` changes | Resolved |

---

# 6. Current Project Status

### Completed

- [x] Baseline video model evaluation
- [x] FaceForensics++ test dataset preparation
- [x] External AI-generated video testing
- [x] Baseline metrics calculation
- [x] Reusable Sentinel analysis pipeline
- [x] AMMP integration
- [x] Temporal analysis integration
- [x] SVIM audio analysis integration
- [x] DTFE trust-score fusion integration
- [x] Full pipeline test on an audio-enabled AI-generated video
- [x] Experiment documentation

### In Progress

- [ ] Improving deepfake detection performance
- [ ] Evaluating the complete pipeline across a larger and more diverse test set
- [ ] Further validation and tuning of multimodal trust-score fusion

### Pending

- [ ] Comprehensive evaluation of the full Sentinel pipeline
- [ ] Comparison of baseline video-only performance against multimodal Sentinel performance
- [ ] Further model/pipeline improvements based on evaluation results
- [ ] Final dashboard and project-level validation

---

# 7. Plan for Next Week

- Run the complete Sentinel pipeline across the available real, FaceForensics++ fake, and external AI-generated videos.
- Compare video-only baseline results with multimodal DTFE results.
- Analyze false positives and false negatives from the evaluation.
- Investigate possible improvements to video, temporal, and audio scoring.
- Expand the evaluation dataset with additional AI-generated and manipulated videos where possible.
- Integrate validated pipeline changes into the Sentinel dashboard.

---

# 8. Guide Remarks

**Guide:**  

**Date:**  

**Remarks:**  

---
