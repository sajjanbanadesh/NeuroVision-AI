# 🧠 NeuroVision AI

AI-powered Alzheimer Disease Prediction System using Deep Learning and MRI Scan Analysis.

---

## 📌 Overview

NeuroVision AI is a web-based Deep Learning application that predicts Alzheimer Disease stages from MRI brain scan images.

The project uses:
- TensorFlow
- Flask
- CNN / MobileNetV2
- HTML/CSS
- MRI Image Processing

Users can upload MRI images through the website and instantly receive:
- Predicted Alzheimer stage
- Confidence score
- Class probabilities

---

# 🚀 Features

✅ MRI Image Upload  
✅ Alzheimer Disease Prediction  
✅ Confidence Percentage  
✅ Deep Learning Model  
✅ Modern Responsive UI  
✅ Flask Web Application  
✅ Real-time Prediction  
✅ Class Probability Visualization  

---

# 🧠 Alzheimer Classes

The model predicts 4 classes:

1. MildDemented
2. ModerateDemented
3. NonDemented
4. VeryMildDemented

---

# 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- Flask
- NumPy
- Pillow (PIL)
- HTML5
- CSS3

---

# 📂 Project Structure

```bash
NeuroVision-AI/
│
├── app.py
├── train.py
├── predict.py
├── alzheimer_model.h5
├── requirements.txt
├── Procfile
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html
│
├── uploads/
│
└── dataset/
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/sajjanbanadesh/neurovision-ai.git
```

---

## 2️⃣ Move into Project Folder

```bash
cd neurovision-ai
```

---

## 3️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Flask Application

```bash
python app.py
```

Open browser:

```bash
http://127.0.0.1:5000
```

---

# 🧪 Train Model

To train model again:

```bash
python train.py
```

---

# 🔍 Predict Using Terminal

```bash
python predict.py
```

---

# 📊 Model Information

- Model Type: MobileNetV2 + Transfer Learning
- Input Size: 224x224
- Framework: TensorFlow/Keras
- Output: 4 Alzheimer Classes

---

# 🌐 Deployment

This project can be deployed using:
- Render
- Railway
- AWS
- Heroku
- PythonAnywhere

---

# 👨‍💻 Developer

## Banadesh Sajjan

📧 Email: Sajjanbanadesh@gmail.com

🔗 LinkedIn:  
https://www.linkedin.com/in/banadesh-sajjan-6a8833212

💻 GitHub:  
https://github.com/sajjanbanadesh

---

# 📜 License

This project is for educational and research purposes.

---

# ⭐ Future Improvements

- Better MRI Accuracy
- EfficientNet Integration
- Doctor Dashboard
- PDF Report Generation
- Patient History Tracking
- Cloud Deployment
- Authentication System

---

# ❤️ Acknowledgement

Special thanks to:
- TensorFlow
- Flask
- Open Source Community
- Medical Imaging Research Community
