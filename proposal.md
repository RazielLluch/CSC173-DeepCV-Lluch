# CSC173 Deep Computer Vision Project Proposal
**Student:** Josiah Raziel, 2022-0834  
**Date:** 12/14/2025


## 1. Project Title 
Gamer ErgoVision: YOLO-Based Detection of Sitting Posture and Screen Distance for PC Gamers


## 2. Problem Statement
Many PC gamers spend long hours in front of their computers using improvised or suboptimal desk and chair setups, often without awareness of how their posture and viewing distance affect long-term comfort and health. Poor sitting posture and being too close to the monitor can contribute to neck and back pain, eye strain, and reduced performance over time, especially for students who also use the same setup for studying. This project aims to build a vision-based tool that detects a gamer’s sitting posture and approximate head-to-screen distance from a side-view webcam image, then classifies the ergonomic risk level. The system is intended as a lightweight assistant to help students and gamers quickly evaluate whether their current setup is ergonomically safe or likely to cause problems during long sessions.


## 3. Objectives
- Develop and train a YOLO-based deep computer vision model that can detect the gamer’s upper body and monitor, and classify sitting posture into a small set of categories (e.g., neutral, forward-leaning, slouching) with a target accuracy of at least 80% on a held-out test set.
- Implement a complete pipeline including dataset preparation, model training, validation, and evaluation in PyTorch, using existing annotated posture and screen-detection datasets.
- Prototype a simple “ergonomic risk” indicator that combines predicted posture class and relative head-to-screen distance to categorize risk as low, medium, or high for long gaming sessions.


## 4. Dataset Plan
- **Source:**
  - Sitting posture: Roboflow “Sitting Posture Classification” with labeled images of good vs bad.
  - Screen detection: “Screen Detection YOLOv8” dataset from Mendeley, containing annotated images of phone, laptop, tablet, TV, and desktop monitor screens for object detection.

- **Classes (tentative):**
  - Posture classes (from posture dataset, mapped as needed):
    - Neutral / ergonomic
    - Forward-lean / hunched
    - Slouch / rounded back
    - Reclined / away (if available)  
  - Screen classes:
    - Desktop monitor (primary target)
    - Laptop screen (optional, if present in dataset)

- **Acquisition:**
  - Download the Roboflow posture datasets in YOLO-compatible format (classification or detection) and the Screen Detection YOLOv8 dataset for monitor bounding boxes. [Sitting Posture - ezkda(Roboflow)](https://universe.roboflow.com/dataset-sqm0h/sitting-posture-ezkda)
  - Optionally collect a small number of side-view “gamer at desk” images locally (with faces cropped or blurred) to use as an extra test set to evaluate model performance in a realistic student/gamer setup.


## 5. Technical Approach
- **Architecture sketch:**
  - Stage 1 (Detection): YOLO-based detector to locate the gamer’s upper body/head region and the monitor/screen in each image using bounding boxes.
  - Stage 2 (Posture classification): Use either a YOLO classification head or a small CNN / classifier on the detected person crop to predict posture category (neutral, forward-lean, slouch, etc.).
  - Geometry module: Compute relative head-to-screen distance proxy using bounding box sizes and positions, then map posture + distance to an overall ergonomic risk label.

- **Model:**
  - Ultralytics YOLOv8 (small model variant) as the core detector, conceptually aligned with CSPDarknet-style backbones discussed in class, fine-tuned on sitting-posture and screen-detection datasets.

- **Framework:**
  - PyTorch via the Ultralytics YOLOv8 library for training, validation, and inference, with custom Python code for posture risk logic.

- **Hardware:**
  - Google Colab GPU (or equivalent cloud GPU environment) for training and experimentation.
  - Personal Laptop(RTX 4050 6GB) for data preprocessing, configuration, and running lighter inference demos.


## 6. Expected Challenges & Mitigations
- **Challenge:** Domain gap between public posture datasets and actual gamer setups (different chairs, desks, camera angles).  
  - **Mitigation:** Use data augmentation (cropping, brightness/contrast, small rotations) and include a small, diverse set of local “gamer desk” images for evaluation; if necessary, fine-tune the model on a small subset of these images.

- **Challenge:** Accurately estimating head-to-screen distance from a single 2D image.  
  - **Mitigation:** Use relative measures (head box size vs monitor box size and spacing) instead of absolute distance, and categorize only into coarse bins (too close / acceptable / far) rather than exact centimeters.

- **Challenge:** Class overlap between posture categories (e.g., subtle difference between “forward-lean” and “slouch”).  
  - **Mitigation:** Limit the number of posture classes to a small, clearly separable set; experiment with combining similar classes and report confusion matrix analysis to understand typical misclassifications.
