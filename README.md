# LSK-LA-RotDet

**Dynamic Receptive Field Meets Local Attention for Rotated Small Object Detection in Remote Sensing Imagery**

## 🧠 Overview

LSK-LA RotDet is a lightweight yet high-performance rotated object detection framework designed for remote sensing imagery. It addresses key challenges such as:

- Dense small objects  
- Large scale variations  
- Arbitrary object orientations  

The model integrates:

- **LSK-DRFNet** (Dynamic Receptive Field Backbone)
- **LA-PAFPN** (Local Attention Feature Pyramid Network)

Achieving strong performance on benchmark datasets while maintaining efficiency.

---

## 🚀 Highlights

- 🔍 Dynamic receptive field for multi-scale feature extraction  
- 🎯 Local attention mechanism for precise localization  
- ⚡ Lightweight architecture for real-time inference  
- 📈 State-of-the-art performance on DOTA & HRSC2016  

---


## 🏗️ Architecture

```
Input Image
↓
LSK-DRFNet (Backbone)
↓
LA-PAFPN (Neck)
↓
Detection Head (Rotated BBox)
```


### Key Components

#### 1. LSK-DRFNet
- Large kernel convolution
- Dynamic receptive field adjustment
- Enhanced small object perception

#### 2. LA-PAFPN
- Bidirectional feature fusion
- Local Importance-based Attention (LIA)
- Improved boundary and orientation modeling

---

## 📊 Results

### DOTA-v1.0

| Model | mAP (%) | FPS |
|------|--------|-----|
| Baseline | 74.6 | 21.7 |
| Ours | **77.8** | 28.9 |

### HRSC2016

| Model | mAP (%) |
|------|--------|
| Baseline | 75.81 |
| Ours | **77.42** |

---
## 👨‍💻 Author

- **Ziyi Jia**  

Xi’an Jiaotong-Liverpool University

---

## 📬 Contact

- 📧 jiaziyi2005@126.com

If you have any feedback or ideas for collaboration, please feel free to reach out! 😊
