# Case Study: Cross-Architecture Evaluation on Generative Media (Gemini / Veo)

## Overview
- **Evaluated Media:** `test_videos/external_ai/gemini_fake_01.mp4.mp4`
- **Source Generator:** Google Gemini / DeepMind Veo (Holistic Video Diffusion)
- **Primary Model Evaluated:** EfficientNet-B0 (Trained on Celeb-DF v2, 85.33% benchmark accuracy)
- **Pipeline Execution:** Full Multimodal Mode (AMMP + AFCP + OpenCV Optical Flow + SVIM + DTFE)

## Diagnostic Results

| Modality / Component | Value / Score | Diagnostic Note |
| :--- | :--- | :--- |
| **Spatial Analysis (AFCP)** | 100.0 (99.95% Real) | 10/10 sampled frames predicted as REAL. The spatial CNN fails to detect holistic diffusion synthesis due to the absence of face-swap boundary seams. |
| **Temporal Consistency** | 52.0 / 100 | OpenCV optical flow registered a motion consistency score of 39/100, flagging non-rigid frame transitions. |
| **Spectral Voice (SVIM)** | 46.0 / 100 | Detected synthetic audio characteristics: flat MFCC dynamics and compressed spectral contrast diversity. |
| **Video Score (Combined)** | 80.8 / 100 | $0.6 \times 100.0 + 0.4 \times 52.0 = 80.8$ |
| **DTFE Final Trust Score** | **66.9 / 100** | $0.6 \times 80.8 + 0.4 \times 46.0 = 66.88 \approx 66.9$ |
| **Final Classification** | **Suspicious / Inconclusive** | Downgraded from Authentic (99.95) to Suspicious (66.9) due to temporal and acoustic penalties. |

## Technical Conclusion
Single-frame spatial backbones trained on facial replacement datasets are vulnerable to holistic text-to-video generative architectures. Sentinel's Dynamic Trust Fusion Engine (DTFE) successfully corrects for this single-modality false negative by penalizing the media's trust score using temporal flow instability and audio spectral anomalies.
```[cite: 1, 4, 5, 8, 9, 16, 17, 20]

---

### Task 3: Clean Up Intermediate Temporary Files

To prevent disk bloat from the 60 extracted AMMP frames and audio wav files generated during the pipeline runs[cite: 5, 18], open PowerShell and run:

```powershell
Remove-Item -Recurse -Force outputs\pipeline\*