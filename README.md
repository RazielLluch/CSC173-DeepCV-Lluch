# Gamer ErgoVision: YOLO-Based Detection of Sitting Posture and Screen Distance for PC Gamers
**CSC173 Intelligent Systems Final Project**  
*Mindanao State University - Iligan Institute of Technology*  
**Student:** Josiah Raziel, 2022-0834  

**Semester:** AY 2025-2026 Sem 1 
[![Python](https://img.shields.io/badge/Python-3.13.5-blue)](https://python.org) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)](https://pytorch.org)

## Abstract
This project proposes Gamer ErgoVision, a deep computer vision system that analyzes a PC gamer’s sitting posture and approximate head-to-screen distance to estimate ergonomic risk during long gaming or study sessions. Many students and gamers spend hours in front of a monitor using improvised desk and chair setups, which can lead to poor posture, neck and back pain, and eye strain over time. To address this, the system uses existing sitting posture datasets from Roboflow and a screen detection dataset (Screen Detection YOLOv8) to avoid heavy manual data collection.​

The technical approach fine-tunes a YOLOv8-based detector in PyTorch to locate the gamer’s upper body/head and monitor from a side-view webcam image, then classifies posture into categories such as neutral, forward-leaning, and slouched. Relative geometry between the head and screen bounding boxes provides a proxy for viewing distance, which is combined with the posture class to derive a coarse ergonomic risk label (low/medium/high). Expected results include achieving at least 80% classification accuracy on posture categories and producing intuitive risk feedback on sample gamer setups. The main contributions are: (1) adapting posture and screen-detection datasets to a gamer-centered ergonomics context, and (2) demonstrating how YOLO-based detection plus simple geometric reasoning can provide practical, privacy-conscious posture guidance for everyday users.[web:25][web:41]

## Table of Contents
- [Introduction](#introduction)
- [Related Work](#related-work)
- [Methodology](#methodology)
- [Experiments & Results](#experiments--results)
- [Discussion](#discussion)
- [Ethical Considerations](#ethical-considerations)
- [Conclusion](#conclusion)
- [Installation](#installation)
- [References](#references)

## Introduction
### Problem Statement
Many PC gamers spend long hours in front of their computers using improvised or suboptimal desk and chair setups, often without awareness of how their posture and viewing distance affect long-term comfort and health. Poor sitting posture and being too close to the monitor can contribute to neck and back pain, eye strain, and reduced performance over time, especially for students who also use the same setup for studying. This project aims to build a vision-based tool that detects a gamer’s sitting posture and approximate head-to-screen distance from a side-view webcam image, then classifies the ergonomic risk level. The system is intended as a lightweight assistant to help students and gamers quickly evaluate whether their current setup is ergonomically safe or likely to cause problems during long sessions.

### Objectives
- Develop and train a YOLO-based deep computer vision model that can detect the gamer’s upper body and monitor, and classify sitting posture into a small set of categories (e.g., neutral, forward-leaning, slouching) with a target accuracy of at least 80% on a held-out test set.
- Implement a complete pipeline including dataset preparation, model training, validation, and evaluation in PyTorch, using existing annotated posture and screen-detection datasets.
- Prototype a simple “ergonomic risk” indicator that combines predicted posture class and relative head-to-screen distance to categorize risk as low, medium, or high for long gaming sessions.

![Problem Demo](images/problem_example.gif)

## Related Work
- 

## Methodology
### Dataset
- Source: [Sitting Posture - ezkda(Roboflow)](https://universe.roboflow.com/dataset-sqm0h/sitting-posture-ezkda)
- Split: 70/20/10 train/val/test
- Preprocessing: 

### Architecture
![Model Diagram](images/architecture.png)
- Backbone: [e.g., CSPDarknet53]
- Head: [e.g., YOLO detection layers]
- Hyperparameters: Table below

| Parameter | Value |
|-----------|-------|
| Batch Size | 16 |
| Learning Rate | 0.01 |
| Epochs | 100 |
| Optimizer | SGD |

### Training Code Snippet
train.py excerpt
model = YOLO('yolov8n.pt')
model.train(data='dataset.yaml', epochs=100, imgsz=640)


## Experiments & Results
### Metrics
| Model | mAP@0.5 | Precision | Recall | Inference Time (ms) |
|-------|---------|-----------|--------|---------------------|
| Baseline (YOLOv8n) | 85% | 0.87 | 0.82 | 12 |
| **Ours (Fine-tuned)** | **92%** | **0.94** | **0.89** | **15** |

![Training Curve](images/loss_accuracy.png)

### Demo
![Detection Demo](demo/detection.gif)
[Video: [CSC173_YourLastName_Final.mp4](demo/CSC173_YourLastName_Final.mp4)] [web:41]

## Discussion

## Ethical Considerations
- Bias: Dataset skewed toward plastic/metal; rural waste underrepresented
- Privacy: No faces in training data
- Misuse: Potential for surveillance if repurposed [web:41]

## Conclusion
[Key achievements and 2-3 future directions, e.g., Deploy to Raspberry Pi for IoT.]

## Installation
1. Clone repo: `git clone https://github.com/RazielLluch/CSC173-DeepCV-Lluch`
2. Install deps: `pip install -r requirements.txt`
<!-- 3. Download weights: See `models/` or run `download_weights.sh` -->

**requirements.txt:**


## References
[1] 

## GitHub Pages
View this project site: [https://jjmmontemayor.github.io/CSC173-DeepCV-Montemayor/](https://jjmmontemayor.github.io/CSC173-DeepCV-Montemayor/) [web:32]