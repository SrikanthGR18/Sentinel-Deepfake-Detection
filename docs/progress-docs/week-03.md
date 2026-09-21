# Sentinel – Weekly Progress Diary

## Week 03

**Period:** 7 September 2026 – 13 September 2026

---

# 1. Weekly Objectives

- Investigate the training and dataset configuration of the video detection model.
- Verify the class-label mapping and training pipeline.
- Retrain the EfficientNet-B0 model using the prepared Celeb-DF-v2 dataset.
- Evaluate the trained model on training and validation data.

---

# 2. Team Progress

## Srikanth

### Work Done

- Reviewed the `dataset.py` implementation to verify the REAL/FAKE class mapping.
- Confirmed that:
  - **Class 0 = FAKE**
  - **Class 1 = REAL**
- Reviewed `train_video.py` to understand the EfficientNet-B0 training configuration.
- Verified the use of:
  - Celeb-DF-v2 dataset
  - 80% training / 20% validation video split
  - 10 sampled frames per video
  - 224×224 input images
  - CrossEntropyLoss
  - Adam optimizer
  - Learning rate of `1e-4`
  - Batch size of 32
  - 10 training epochs
- Reviewed `extract_frames.py` and verified the video-level dataset split and frame extraction process.
- Started a fresh training run of the EfficientNet-B0 video model.
- Monitored training and validation loss and accuracy across epochs.

### Challenges Faced

- Training the model required significant processing time.
- Training accuracy continued increasing while validation accuracy fluctuated around the mid-90% range.
- The difference between training and validation performance indicated some degree of model overfitting.

### Solutions

- Monitored validation accuracy after every epoch.
- Used validation accuracy to save the best-performing model checkpoint.
- Continued the complete training run to obtain the best available model for external testing.

### Results

- Completed 10 epochs of EfficientNet-B0 training.
- Final training accuracy: **99.37%**
- Final validation accuracy: **95.04%**
- Best model checkpoint was saved as `weights/best_model.pth`.
- The trained model was ready for evaluation on unseen videos.

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

- [x] Dataset and label mapping verification
- [x] Training pipeline verification
- [x] EfficientNet-B0 retraining
- [x] 10-epoch training run
- [x] Best-model checkpoint generation

### In Progress

- [ ] External video evaluation
- [ ] Error analysis
- [ ] Multimodal validation

### Pending

- [ ] Browser extension integration
- [ ] Final system evaluation

---

# 7. Plan for Next Week

- Evaluate the newly trained model on the complete external test set.
- Compare the new results with the previous baseline.
- Analyze false positives and false negatives.
- Evaluate the model's behaviour on FaceForensics++ and external AI-generated content.
- Determine whether further changes to the video model are required.

---

# 8. Guide Remarks

**Guide:**  

**Date:**  

**Remarks:**  

---