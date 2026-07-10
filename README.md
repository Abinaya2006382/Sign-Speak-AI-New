# 🤟 Sign Speak AI

## 📌 Project Overview

Sign Speak AI is an AI-powered full-stack web application that recognizes sign language hand gestures in real time and converts them into text and speech.

The system uses computer vision and deep learning techniques to detect hand landmarks, classify gestures, and provide voice output to improve communication accessibility.

---

## 🎯 Objective

The main objective of this project is to help bridge communication between deaf/mute communities and others by translating sign language gestures into understandable text and speech.

---

## 🚀 Features

✨ Real-time sign language recognition  
✨ Webcam-based gesture detection  
✨ Hand landmark extraction using MediaPipe  
✨ AI-based gesture classification  
✨ Text conversion  
✨ Voice output support  
✨ Gesture history storage  
✨ Multiple sign gesture support  
✨ Dark futuristic user interface  

---

## 🖐 Supported Gestures

The model currently supports:

1. Hello
2. Bye
3. Good Morning
4. Help
5. I Love You
6. No
7. Sorry
8. Thank You
9. Thumbs Up
10. Thumbs Down
11. Welcome
12. Yes

---

## 🛠 Technologies Used

### Frontend
- HTML
- CSS
- JavaScript
- Bootstrap

### Backend
- Python
- Flask REST API

### AI / Machine Learning
- TensorFlow
- Keras
- OpenCV
- MediaPipe

### Database
- SQLite

---

## 🏗 Project Structure

```
Sign-Speak-AI
│
├── ai_engine
│   ├── gesture_model.py
│   ├── hand_detection.py
│   ├── train_model.py
│   └── dataset
│
├── backend
│   ├── app.py
│   ├── routes
│   └── database
│
├── frontend
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── database
│   └── sign_speak.db
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Abinaya2006382/Sign-Speak-AI.git
```

Go inside the project folder:

```bash
cd Sign-Speak-AI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

Start the Flask backend:

```bash
python backend/app.py
```

Open browser:

```
http://localhost:5000
```

---

## 🧠 Working Process

1. Webcam captures hand gestures
2. MediaPipe detects hand landmarks
3. Landmark data is processed
4. AI model predicts the gesture
5. Gesture is converted into text
6. Voice output speaks the result

---

## 📊 AI Model

The gesture classifier is built using a neural network model.

Input:
- 21 hand landmarks
- 63 coordinate features

Output:
- Gesture class prediction

Training:
- Real collected hand landmark samples
- Data augmentation
- Confidence-based prediction

---

## 📚 Documentation

Project documents:

- ER Diagram
- Use Case Diagram
- SQL Schema
- Database Table List
- Project Details

---

## 🔮 Future Enhancements

- Add more sign language gestures
- Improve accuracy with larger datasets
- Support continuous sentence translation
- Mobile application support
- Multi-language voice output

---

## 👩‍💻 Developed By

**Abinaya**

Sign Speak AI Project
