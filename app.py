from flask import Flask, render_template, request, jsonify
import cv2
import sqlite3
import numpy as np
import onnxruntime as ort
import os
import urllib.request

# ================= CONFIG =================
THRESHOLD = 0.50
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ================= ARC FACE MODEL (PUBLIC LINK) =================
ARC_MODEL_URL = (
    "https://huggingface.co/FoivosPar/Arc2Face/"
    "resolve/da2f1e9aa3954dad093213acfc9ae75a68da6ffd/arcface.onnx"
)
ARC_MODEL_PATH = os.path.join(BASE_DIR, "arcface.onnx")

# Download model once
if not os.path.exists(ARC_MODEL_PATH):
    print("⬇️ Downloading ArcFace model...")
    urllib.request.urlretrieve(ARC_MODEL_URL, ARC_MODEL_PATH)
    print("✅ ArcFace model ready")

# ================= APP =================
app = Flask(__name__)

# ================= DATABASE =================
conn = sqlite3.connect("users.db", check_same_thread=False)
cur = conn.cursor()

# HARD RESET SCHEMA (SAFE & CLEAN)
cur.execute("""
CREATE TABLE IF NOT EXISTS users (
    email TEXT PRIMARY KEY,
    embedding BLOB NOT NULL
)
""")
conn.commit()

# ================= FACE DETECTOR =================
face_detector = cv2.dnn.readNetFromCaffe(
    os.path.join(BASE_DIR, "deploy.prototxt"),
    os.path.join(BASE_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
)

# ================= ARC FACE =================
arcface = ort.InferenceSession(
    ARC_MODEL_PATH,
    providers=["CPUExecutionProvider"]
)
arcface_input_name = arcface.get_inputs()[0].name

# ================= UTILS =================
def cosine_sim(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.dot(a, b))

def detect_face(img):
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104, 177, 123))
    face_detector.setInput(blob)
    detections = face_detector.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.9:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)

            if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                return None

            return img[y1:y2, x1:x2]
    return None

def get_embedding(face):
    face = cv2.resize(face, (112, 112))
    face = face.astype(np.float32) / 255.0
    face = np.expand_dims(face, axis=0)

    emb = arcface.run(None, {arcface_input_name: face})[0][0]
    return emb / np.linalg.norm(emb)

# ================= ROUTES =================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/register-page")
def register_page():
    return render_template("register.html")

@app.route("/register", methods=["POST"])
def register():
    email = request.form.get("email", "").strip().lower()
    print("REGISTER:", email)

    if not email:
        return jsonify({"success": False, "msg": "Email required"})

    cur.execute("SELECT email FROM users WHERE email=?", (email,))
    if cur.fetchone():
        return jsonify({"success": False, "msg": "Email already registered"})

    file = request.files.get("image")
    if not file:
        return jsonify({"success": False, "msg": "No image provided"})

    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    face = detect_face(img)

    if face is None:
        return jsonify({"success": False, "msg": "No face detected"})

    emb = get_embedding(face)

    cur.execute(
        "INSERT INTO users (email, embedding) VALUES (?, ?)",
        (email, emb.tobytes())
    )
    conn.commit()

    print("REGISTERED:", email)
    return jsonify({"success": True, "msg": "Registration successful"})

@app.route("/login", methods=["POST"])
def login():
    email = request.form.get("email", "").strip().lower()
    print("LOGIN:", email)

    if not email:
        return jsonify({"success": False, "msg": "Email required"})

    cur.execute("SELECT embedding FROM users WHERE email=?", (email,))
    row = cur.fetchone()

    if not row:
        return jsonify({"success": False, "msg": "User not found"})

    file = request.files.get("image")
    if not file:
        return jsonify({"success": False, "msg": "No image provided"})

    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    face = detect_face(img)

    if face is None:
        return jsonify({"success": False, "msg": "No face detected"})

    stored_emb = np.frombuffer(row[0], dtype=np.float32)
    emb = get_embedding(face)
    score = cosine_sim(emb, stored_emb)

    if score >= THRESHOLD:
        return jsonify({
            "success": True,
            "email": email,
            "similarity": round(score, 3)
        })

    return jsonify({"success": False, "msg": "Face does not match"})

# ================= RUN =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
