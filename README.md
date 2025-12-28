# Face Recognition Login System

A secure face-based authentication system built using **Python, OpenCV, ArcFace, and Flask**.  
The project supports **face registration via webcam**, stores facial embeddings securely, and allows users to **log in using face recognition** with cosine similarity matching.

---

## 🚀 Features

- Face registration using webcam (CLI-based)
- Face login using facial recognition
- ArcFace-based facial embeddings (high accuracy)
- OpenCV DNN face detection
- SQLite database for user data
- Flask-based backend for login
- Secure password hashing (bcrypt)

---

## 🛠 Tech Stack

- **Python**
- **OpenCV**
- **ONNX Runtime**
- **ArcFace**
- **Flask**
- **SQLite**
- **NumPy**
- **Scikit-learn**

---

## 📁 Project Structure

Face-Recognition-Login-System/
│
├── app.py # Flask backend (login)
├── app_cli.py # CLI-based face registration
├── deploy.prototxt # Face detector config
├── requirements.txt
├── README.md
├── .gitignore
│
├── templates/
│ └── index.html
│
├── static/
│ ├── script.js
│ └── style.css

yaml


## 📦 Installation

### 1️⃣ Clone the repository
git clone https://github.com/AnujBhatia7471/Face-Recognition-Login-System.git
cd Face-Recognition-Login-System

2️⃣ Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate   # Windows

3️⃣ Install dependencies
pip install -r requirements.txt
🧠 Model Downloads (Required)
This project uses pretrained models which are not included in the repository.

🔹 ArcFace (Face Recognition)
Download an ArcFace ONNX model from:


https://github.com/deepinsight/insightface/tree/master/model_zoo
Rename the file to:
arcface.onnx
Place it in the project root directory.

🔹 OpenCV Face Detector (Caffe)
Download the model from:
https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
Place it in the project root directory.

deploy.prototxt is already included in the repository.

🧑‍💻 Usage
1️⃣ Register a new user (Face Enrollment)
python app_cli.py
Enter email and password

Press SPACE to capture face samples

Press ESC to cancel

2️⃣ Start the Flask server
bash
Copy code
python app.py
Open browser:
http://127.0.0.1:5000

🔐 Security Notes
Facial embeddings are stored securely as vectors
Passwords are hashed using bcrypt
Database and model files are ignored via .gitignore

📌 Notes
Ensure no other application is using the webcam
Best results in good lighting
Supports single-face detection per frame

📄 License
This project is for educational and learning purposes.

👤 Author
Anuj Bhatia
GitHub: https://github.com/AnujBhatia7471
