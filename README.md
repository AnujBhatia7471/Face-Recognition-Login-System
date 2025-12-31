# Face Recognition Login System
The **Face Recognition Login System** is a browser-based facial authentication application developed using **Python, OpenCV, ArcFace, and Flask**.  
The system allows users to **register and log in using their face via a webcam**, eliminating the need for traditional password-based authentication.

This project demonstrates practical implementation of **computer vision, deep learning inference, backend development, and web-based camera integration**, making it suitable for academic use and professional portfolio showcasing.


## Features
- Browser-based face registration (no CLI or terminal interaction)
- Face-based login authentication using webcam
- ArcFace ONNX model for high-accuracy facial embeddings
- Cosine similarity-based face matching
- OpenCV DNN face detection
- SQLite database for secure storage of facial embeddings
- Flask-based REST backend
- Automatic model download at runtime (no large files stored in GitHub)
- Clean and responsive user interface


## Technology Stack
- Python  
- Flask  
- OpenCV  
- ONNX Runtime  
- ArcFace  
- SQLite  
- NumPy  
- HTML, CSS, JavaScript  


## Project Structure
Face-Recognition-Login-System/
│
├── app.py
├── deploy.prototxt
├── res10_300x300_ssd_iter_140000.caffemodel
├── requirements.txt
├── README.md
├── .gitignore
│
├── templates/
│ ├── index.html
│ └── register.html
│
├── static/
│ ├── style.css
│ ├── login.js
│ └── register.js



## Installation
### Clone the Repository
git clone https://github.com/AnujBhatia7471/Face-Recognition-Login-System.git
cd Face-Recognition-Login-System
Create a Virtual Environment (Recommended)
python -m venv venv
venv\Scripts\activate   # Windows
Install Dependencies
pip install -r requirements.txt
Model Handling
The ArcFace ONNX model is not included in the repository due to file size limitations.
The model is automatically downloaded at runtime
No manual setup is required
This approach reflects standard industry deployment practices

**Usage**
Start the Application
python app.py
Open in Browser
http://localhost:5000/

**Application Workflow** 
Face Registration
User enters an email address
Webcam is activated in the browser
Facial image is captured
ArcFace generates a facial embedding
The embedding is stored securely in the database

**Face Login**
User enters a registered email address
Webcam captures a live facial image
A new facial embedding is generated
Cosine similarity is calculated against the stored embedding
Authentication is approved or rejected based on the similarity score

**Security Considerations**
Facial embeddings are stored instead of raw images
No facial images are saved on disk
Single-face detection is enforced per frame
Designed for educational and demonstration purposes

**Notes**
Ensure camera permission is enabled in the browser
Best results are achieved under good lighting conditions
Camera access works on:
http://localhost
Secure HTTPS deployments
Multiple faces in a single frame are not supported

**Deployment**
The application can be deployed on cloud platforms such as Render.
For deployed environments, HTTPS is required to enable browser camera access.

**License**
This project is intended for educational and learning purposes only.

**Author**
Anuj Bhatia
GitHub: https://github.com/AnujBhatia7471

**Summary**
This project presents a complete facial authentication pipeline, integrating computer vision, deep learning inference, backend APIs, and frontend camera handling.
It effectively demonstrates the development of an AI-powered web authentication system.

