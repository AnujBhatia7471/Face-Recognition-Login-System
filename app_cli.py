import cv2
import sqlite3
import bcrypt
import numpy as np
import onnxruntime as ort
import sys
import time

# ================= CONFIG =================
SAMPLES_REQUIRED = 5
FACE_CONFIDENCE = 0.9

# ================= DATABASE =================
conn = sqlite3.connect("users.db")
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    password BLOB NOT NULL,
    embedding BLOB NOT NULL
)
""")
conn.commit()

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
input_name = arcface.get_inputs()[0].name

# ================= FACE UTILS =================
def detect_face(img):
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(
        img, 1.0, (300, 300), (104, 177, 123)
    )
    face_detector.setInput(blob)
    detections = face_detector.forward()

    for i in range(detections.shape[2]):
        if detections[0, 0, i, 2] > FACE_CONFIDENCE:
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
    emb = arcface.run(None, {input_name: face})[0][0]
    return emb / np.linalg.norm(emb)

# ================= REGISTRATION INPUT =================
email = input("Email (unique): ").strip().lower()
password = bcrypt.hashpw(
    input("Password: ").encode(),
    bcrypt.gensalt()
)

embeddings = []

# ================= CAMERA INIT (CORRECT WAY) =================
cv2.setNumThreads(1)

cam = cv2.VideoCapture(0)  # ← NO BACKEND, NO FOURCC

time.sleep(0.5)

if not cam.isOpened():
    print("❌ Camera not accessible")
    sys.exit(1)

print(f"\nCapture {SAMPLES_REQUIRED} face samples")
print("Press SPACE to capture | ESC to quit\n")

# ================= CAPTURE LOOP =================
while len(embeddings) < SAMPLES_REQUIRED:
    ret, frame = cam.read()

    if not ret:
        print("⚠ Frame grab failed, retrying...")
        time.sleep(0.1)
        continue   # 🔴 DO NOT show garbage frame

    face = detect_face(frame)

    cv2.imshow("Registration", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        print("\n❌ Registration cancelled")
        cam.release()
        cv2.destroyAllWindows()
        sys.exit(0)

    if key == 32:  # SPACE
        if face is None:
            print("⚠ No face detected")
            time.sleep(0.3)
            continue

        embeddings.append(get_embedding(face))
        print(f"Captured {len(embeddings)}/{SAMPLES_REQUIRED}")
        time.sleep(0.5)

# ================= CLEAN RELEASE =================
cam.release()
cv2.destroyAllWindows()

# ================= VALIDATION =================
if len(embeddings) != SAMPLES_REQUIRED:
    print("❌ Insufficient samples captured")
    sys.exit(1)

avg_embedding = np.mean(embeddings, axis=0)
avg_embedding /= np.linalg.norm(avg_embedding)

# ================= SAVE =================
try:
    cur.execute(
        "INSERT INTO users (email, password, embedding) VALUES (?, ?, ?)",
        (email, password, avg_embedding.tobytes())
    )
    conn.commit()
except sqlite3.IntegrityError:
    print("❌ Email already registered")
    sys.exit(1)

print("\n✅ Registration successful")
