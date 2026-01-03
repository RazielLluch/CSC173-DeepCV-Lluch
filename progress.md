# CSC173 Deep Computer Vision Project Progress Report
**Student:** Josiah Raziel S. Lluch, 2022-0834  
**Date:** Dec 19, 2025  
**Repository:** [https://github.com/RazielLluch/CSC173-DeepCV-Lluch](https://github.com/RazielLluch/CSC173-DeepCV-Lluch)  


## 📊 Current Status
| Milestone | Status | Notes |
|-----------|--------|-------|
| Dataset Preparation | ✅ Completed | 1595 images downloaded/preprocessed |
| Initial Training | ✅ Completed | 30 epochs completed |
| Baseline Evaluation | ✅ Completed | Model performs good when hands on lap, perfoms bad when hands on desk |
| Model improvement | ⏳ In Progress | Implementing Pose Extraction alongside appearance for classification |
| Model Training | ⏳ Not Started | Planned for tomorrow |

## 1. Dataset Progress
- **Total images:** 1595
- **Train/Val/Test split:** 1314/187/94
- **Classes implemented:** 3 classes: goodposture, backwardleanbadposture, forwardleanbadposture
- **Preprocessing applied:** Resize(640), normalization, augmentation (flip, rotate, brightness)

**Sample data preview:**

| Good Posture | Foward Lean Bad Posture | Backward Lean Bad Posture |
| --- | --- | --- |
| ![Dataset Sample](yolo_dataset/test/goodposture/good_posture-341-_jpg.rf.2bd8256929c5e903d51a503912758015.jpg) | ![Dataset Sample](yolo_dataset/test/backwardbadposture/backward_lean_bad_posture-36-_jpg.rf.0f38206872303fad4e879a52aba72a60.jpg) | ![Dataset Sample](yolo_dataset/test/forwardbadposture/forward_lean_bad_posture-46-_jpg.rf.c0fed734bcbe0416b767e59d8ec0bf13.jpg) | 


## 2. Training Progress

**Training Curves (so far)**
![Training and Validation Metrics](v1\results\posture_training\training_history.png)


**Current Metrics:**
|   | precision | recall | f1-score | support |
|---|-----------|--------|----------|---------|
| Backward Lean | 1.0 | 0.86 | 0.93 | 58 |
| Forward Lean | 0.96 | 1.0 | 0.98 | 44 |
| Good | 0.91 | 0.98 | 0.94 | 86 |
| | | | | |
| accuracy | | | 0.95 | 188 |
| macro avg | 0.96 | 0.95 | 0.95 | 188 |
| weighted avg | 0.95 | 0.95 | 0.95 | 188 |

## 3. Challenges Encountered & Solutions
| Issue | Status | Resolution |
|-------|--------|------------|
| Slow single epoch training | ✅ Fixed | increased batch size 16 -> 128 (RTX 4050 6GB) |
| Class imbalance | ⏳ Planned | Re-train with new pipeline |
| Wrong classification with different hand positions | ⏳ Planned | Implement dual architecture using pose estimation |

## 4. Next Steps (Before Final Submission)
- [ ] Create new pipeline with pose estimation
- [ ] Complete training 30 epochs
- [ ] Baseline comparison (vs. initial model)
- [ ] Record 5-min demo video
- [ ] Write complete README.md with results