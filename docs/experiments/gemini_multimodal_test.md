# Gemini AI Video – Full Sentinel Pipeline Test

## Ground Truth

Actual classification: FAKE
Source: Gemini-generated video

## Results

Video model: 100.00% REAL
Temporal score: 52.00
MFCC score: 49.00
Acoustic pattern score: 39.00
Audio score: 46.00

DTFE final score: 66.90
Classification: Suspicious
Confidence: Medium

## Observation

The video classification model alone incorrectly classified the
AI-generated video as Authentic.

The multimodal Sentinel pipeline incorporated temporal and audio
evidence through DTFE and classified the video as Suspicious.

## Conclusion

The test demonstrates the intended purpose of multimodal trust-score
fusion, although additional videos are required before drawing
conclusions about overall system accuracy.