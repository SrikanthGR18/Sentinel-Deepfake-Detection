# Sentinel Baseline Evaluation

## Objective

Evaluate the current video deepfake detection model on
unseen FaceForensics++ videos and an externally generated AI video.

## Model

Current Sentinel EfficientNet-based video classification model.

The model was trained using the project's existing training dataset.

## Test Dataset

- 5 FaceForensics++ original videos
- 5 FaceForensics++ Deepfake videos
- 1 external AI-generated video

## Results

| Metric | Result |
|---|---:|
| Total Videos | 11 |
| Correct Predictions | 6 |
| Accuracy | 54.55% |
| Precision | 66.67% |
| Recall | 33.33% |
| F1 Score | 44.44% |

## Confusion Matrix

| Actual / Predicted | REAL | FAKE |
|---|---:|---:|
| REAL | 4 | 1 |
| FAKE | 4 | 2 |

## Observations

1. The model correctly classified 4/5 real FaceForensics++ videos.
2. The model correctly classified only 2/5 FaceForensics++ deepfake videos.
3. The external Gemini-generated video was incorrectly classified as real.
4. The model currently shows poor generalization to unseen manipulation sources.
5. False negatives are the major problem.
6. The low recall of 33.33% indicates that many fake videos are being classified as authentic.

## Conclusion

The current EfficientNet spatial model provides a baseline but is
not sufficient as a standalone deepfake detection system.

Further improvement is required through additional training data,
temporal analysis, audio analysis, and multimodal trust-score fusion.