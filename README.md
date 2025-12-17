# 👁️ SmartVisionAI: Next-Gen Distracted Driver Detection

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-blue?style=for-the-badge&logo=python)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-green?style=for-the-badge&logo=opencv)](https://opencv.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-yellow?style=for-the-badge&logo=python)](https://www.python.org/)

> **"Executive-Ready AI Intelligence."**  
> An advanced computer vision system that goes beyond simple detection. SmartVisionAI uses **logic-based inference** (e.g., checking if a "Phone" intersects with a "Hand") to accurately identify distracted driving in real-time.

---
## ⛓️‍💥[Streamlit](https://arunsmartvisionai.streamlit.app/) 


## 🚀 Key Features

### 1. 🌌 "Onyx & Neon" Holographic UI
Designed for high-level presentations, the interface features a **glass-morphism aesthetic** with a dark, futuristic theme. It transforms raw data into a visual command center.

### 2. 🧠 Logic-Based Detection Engine
Standard models often misclassify objects. SmartVisionAI uses a **custom logic layer**:
- **Intersection Analysis:** A "Phone" is only flagged as a threat if it physically overlaps with a "Hand" bounding box.
- **Reduced False Positives:** Drastically improves reliability compared to standard YOLO classification.

### 3. 🎛️ Live Sensitivity Control
Debug in real-time during demos. The sidebar features a **dynamic confidence slider** that lets you adjust the model's sensitivity (0.0 to 1.0) on the fly without restarting the app.

### 4. 🎓 "Under the Hood" Explainer Mode
Built for stakeholders and non-technical audiences (CEOs/Professors).
- **Interactive Expanders:** Every section includes a dropdown explaining *how* the technology works (IoU, Confidence Scores, Neural Networks).
- **Educational Value:** Turns the demo into a learning experience.

---

## 🛠️ Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Core AI** | **YOLOv8** (Ultralytics) | SOTA Object Detection model trained on custom distraction datasets. |
| **Interface** | **Streamlit** | Interactive web dashboard with custom CSS styling. |
| **Vision** | **OpenCV** (cv2) | Real-time frame processing and bounding box logic. |
| **Data** | **Pandas & NumPy** | Matrix operations for IoU calculations and analytics. |
| **Deployment** | **Streamlit Cloud** | Cloud-ready architecture with `requirements.txt` optimization. |

---

## ⚙️ Installation & Setup

1. **Clone the Repository**
