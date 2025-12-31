from flask import Flask, render_template, request, jsonify, session, redirect
import cv2
import sqlite3
import numpy as np
import os
import onnxruntime as ort
import urllib.request
import threading
import sys
import traceback

print("🔥 app.py started")
print("📁 BASE DIR:", os.getcwd())

# ================= APP =================
app = Flask(__name__)
app.secret_key = "super-secret-key"
app.debug = True

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ================= DATABASE =================
def get_db():
    conn = sqlite3.connect("users.db", check_same_thread=False)
    return conn, conn.cursor()

conn, cur = get_db()
cur.execute("""
CREATE TABLE IF NOT EXISTS users (
    email TEXT PRIMARY KEY,
    password TEXT NOT NULL
)
""")
cur.execute("""
CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL,
    embedding BLOB NOT NULL
)
""")
conn.commit()
conn.close()

# ================= VERIFY OPENCV FILES =================
PROTO_PATH = os.path.join(BASE_DIR, "deploy.prototxt")
MODEL_PATH_CAFFE = os.path.join(BASE_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

print("🔍 prototxt exists:", os.path.exists(PROTO_PATH))
print("🔍 caffemodel exists:", os.path.exists(MODEL_PATH_CAFFE))

if not os.path.exists(PROTO_PATH) or not os.path.exists(MODEL_PATH_CAFFE):
    print("❌ OpenCV model files missing — Render WILL FAIL")
else:
    print("✅ OpenCV model files found")

# ================= FACE DETECTOR =================
face_detector = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH_CAFFE)

# ================= ARC FACE (HUGGINGFACE SAFE LOAD) =================
ARC_URL = (
    "https://huggingface.co/FoivosPar/Arc2Face/resolve/"
    "da2f1e9aa3954dad093213acfc9ae75a68da6ffd/arcface.onnx"
)
ARC_PATH = os.path.join(BASE_DIR, "arcface.onnx")

arcface = None
arc_input_name = None
arc_lock = threading.Lock()
THRESHOLD = 0.50

def ensure_arcface():
    if not os.path.exists(ARC_PATH):
        print("⬇️ Downloading ArcFace model...")
        urllib.request.urlretrieve(ARC_URL, ARC_PATH)
        print("✅ ArcFace downloaded")

def get_arcface():
    global arcface, arc_input_name
    with arc_lock:
        if arcface is None:
            ensure_arcface()
            arcface = ort.InferenceSession(
                ARC_PATH,
                providers=["CPUExecutionProvider"]
            )
            arc_input_name = arcface.get_inputs()[0].name
            print("✅ ArcFace input:", arc_input_name)
            print("✅ ArcFace shape:", arcface.get_inputs()[0].shape)
    return arcface

# ================= UTILS =================
def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def detect_face(img):
    if img is None:
        return None

    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(
        img, 1.0, (300, 300), (104, 177, 123)
    )
    face_detector.setInput(blob)
    dets = face_detector.forward()

    for i in range(dets.shape[2]):
        if dets[0, 0, i, 2] > 0.9:
            box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            face = img[y1:y2, x1:x2]
            if face.size:
                return face
    return None

def get_embedding(face):
    face = cv2.resize(face, (112, 112))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype(np.float32) / 255.0
    face = np.expand_dims(face, axis=0)  # NHWC

    session = get_arcface()
    emb = session.run(None, {arc_input_name: face})[0][0]
    return emb / np.linalg.norm(emb)

# ================= ROUTES =================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/register", methods=["POST"])
def register():
    try:
        email = request.form.get("email", "").lower()
        password = request.form.get("password", "")
        image = request.files.get("image")

        if not email or not password or not image:
            return jsonify(success=False, msg="Missing data")

        img = cv2.imdecode(
            np.frombuffer(image.read(), np.uint8),
            cv2.IMREAD_COLOR
        )

        # ⬇️ Resize for Render memory safety
        img = cv2.resize(img, (640, 480))

        face = detect_face(img)
        if face is None:
            return jsonify(success=False, msg="No face detected")

        emb = get_embedding(face)

        conn, cur = get_db()
        cur.execute(
            "INSERT OR IGNORE INTO users VALUES (?, ?)",
            (email, password)
        )
        cur.execute(
            "INSERT INTO embeddings (email, embedding) VALUES (?, ?)",
            (email, emb.tobytes())
        )
        conn.commit()
        conn.close()

        return jsonify(success=True, msg="Registered")

    except Exception as e:
        print("🔥 REGISTER ERROR")
        traceback.print_exc()
        return jsonify(success=False, error=str(e)), 500

@app.route("/login/face", methods=["POST"])
def face_login():
    try:
        email = request.form.get("email")
        image = request.files.get("image")

        img = cv2.imdecode(
            np.frombuffer(image.read(), np.uint8),
            cv2.IMREAD_COLOR
        )
        img = cv2.resize(img, (640, 480))

        face = detect_face(img)
        if face is None:
            return jsonify(success=False, msg="No face detected")

        emb = get_embedding(face)

        conn, cur = get_db()
        cur.execute("SELECT embedding FROM embeddings WHERE email=?", (email,))
        rows = cur.fetchall()
        conn.close()

        for r in rows:
            stored = np.frombuffer(r[0], dtype=np.float32)
            if cosine_sim(emb, stored) >= THRESHOLD:
                session["user"] = email
                return jsonify(success=True, msg="Login success")

        return jsonify(success=False, msg="Face mismatch")

    except Exception as e:
        print("🔥 LOGIN ERROR")
        traceback.print_exc()
        return jsonify(success=False, error=str(e)), 500

# ================= MAIN =================
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
