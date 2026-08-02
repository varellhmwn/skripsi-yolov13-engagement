# Real-time Smoothed Inference Summary

- **Model Weights:** `D:\varell\college\tugas\semester 7\projek skripsi1\skripsi_yolov13_engagement\runs\yolov13_master_combined_v2\weights\best.pt`
- **Source:** `0`

## Smoothing Parameters
- `window_size`: 30
- `min_vote_ratio`: 0.4
- `min_avg_confidence`: 0.5

## Execution Stats
- Total frames processed: 4930
- Frames with `no_face`: 100
- Frames with `neutral`: 183
- Hard samples collected: 7

## Raw Prediction Distribution
- engaged: 1509
- confused: 784
- bored: 598
- frustrated: 1939

## Stable Prediction Distribution
- engaged: 1583
- confused: 772
- bored: 430
- frustrated: 1862

## Hard Samples per Kelas (Kumulatif)
- engaged: 0 citra + label
- confused: 6 citra + label
- bored: 2 citra + label
- frustrated: 0 citra + label