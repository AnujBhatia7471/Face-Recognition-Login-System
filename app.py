from flask import Flask, render_template, request, jsonify
import cv2
import sqlite3
import numpy as np
import onnxruntime as ort
from sklearn.metrics.pairwise import cosine_similarity

# ================= CONFIG =================
THRESHOLD = 0.50

# ================= APP =================
app = Flask(__name__)

# ================= DATABASE =================
conn = sqlite3.connect("users.db", check_same_thread=False)
cur = conn.cursor()

# ================= FACE DETECTOR =================
face_detector = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)

# ================= ARC FACE =================
arcface = ort.InferenceSession(
    "arcface.onnx",
    providers=["CPUExecutionProvider"]
)
arcface_input_name = arcface.get_inputs()[0].name

# ================= FACE UTILS =================
def detect_face(img):
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(
        img, 1.0, (300, 300), (104, 177, 123)
    )
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
    face = np.expand_dims(face, axis=0)  # NHWC

    emb = arcface.run(
        None,
        {arcface_input_name: face}
    )[0][0]

    return emb / np.linalg.norm(emb)

# ================= ROUTES =================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login", methods=["POST"])
def login():
    email = request.form.get("email")

    # 🔒 ONLY THIS USER IS FETCHED (1:1 VERIFICATION)
    cur.execute(
        "SELECT embedding FROM users WHERE email=?",
        (email,)
    )
    row = cur.fetchone()

    if not row:
        return jsonify({
            "success": False,
            "msg": "User not found"
        })

    stored_embedding = np.frombuffer(row[0], dtype=np.float32)

    # Read uploaded image
    file = request.files.get("image")
    if not file:
        return jsonify({
            "success": False,
            "msg": "No image provided"
        })

    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    face = detect_face(img)
    if face is None:
        return jsonify({
            "success": False,
            "msg": "No face detected"
        })

    emb = get_embedding(face)
    score = cosine_similarity(
        [emb],
        [stored_embedding]
    )[0][0]

    if score >= THRESHOLD:
        return jsonify({
            "success": True,
            "email": email,
            "similarity": round(float(score), 3)
        })
    else:
        return jsonify({
            "success": False,
            "msg": "Face does not match"
        })

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)
