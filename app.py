from flask import Flask, render_template, request, jsonify, session, redirect
import sqlite3
import numpy as np
import os

print("🔥 app.py started (RENDER SAFE VERSION)")

# ================= APP =================
app = Flask(__name__)
app.secret_key = "super-secret-key"
app.debug = False

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

# ================= UTILS =================
def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ================= ROUTES =================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/register-page")
def register_page():
    return render_template("register.html")

@app.route("/dashboard")
def dashboard():
    if not session.get("user"):
        return redirect("/")
    return render_template("dashboard.html", email=session["user"])

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# ================= REGISTER =================
@app.route("/register", methods=["POST"])
def register():
    try:
        data = request.get_json()

        email = data.get("email", "").strip().lower()
        password = data.get("password", "")
        embedding = data.get("embedding")  # list[float]

        if not email or not password or not embedding:
            return jsonify(success=False, msg="Missing data")

        emb = np.array(embedding, dtype=np.float32)

        conn, cur = get_db()

        cur.execute("SELECT COUNT(*) FROM embeddings WHERE email=?", (email,))
        if cur.fetchone()[0] >= 5:
            conn.close()
            return jsonify(success=False, msg="Already registered")

        cur.execute("SELECT email FROM users WHERE email=?", (email,))
        if not cur.fetchone():
            cur.execute(
                "INSERT INTO users (email, password) VALUES (?, ?)",
                (email, password)
            )

        cur.execute(
            "INSERT INTO embeddings (email, embedding) VALUES (?, ?)",
            (email, emb.tobytes())
        )

        conn.commit()
        conn.close()

        return jsonify(success=True, msg="Face sample saved")

    except Exception as e:
        print("❌ REGISTER ERROR:", e)
        return jsonify(success=False, msg="Server error"), 500

# ================= LOGIN (FACE) =================
@app.route("/login/face", methods=["POST"])
def face_login():
    try:
        data = request.get_json()

        email = data.get("email", "").strip().lower()
        embedding = data.get("embedding")

        if not email or not embedding:
            return jsonify(success=False, msg="Missing data")

        emb = np.array(embedding, dtype=np.float32)

        conn, cur = get_db()
        cur.execute("SELECT embedding FROM embeddings WHERE email=?", (email,))
        rows = cur.fetchall()
        conn.close()

        if not rows:
            return jsonify(success=False, msg="User not registered")

        for r in rows:
            stored = np.frombuffer(r[0], dtype=np.float32)
            if cosine_sim(emb, stored) >= 0.50:
                session["user"] = email
                return jsonify(success=True, msg="Login successful")

        return jsonify(success=False, msg="Face does not match")

    except Exception as e:
        print("❌ LOGIN ERROR:", e)
        return jsonify(success=False, msg="Server error"), 500

# ================= LOGIN (PASSWORD) =================
@app.route("/login/password", methods=["POST"])
def password_login():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    conn, cur = get_db()
    cur.execute("SELECT password FROM users WHERE email=?", (email,))
    row = cur.fetchone()
    conn.close()

    if not row or row[0] != password:
        return jsonify(success=False, msg="Invalid credentials")

    session["user"] = email
    return jsonify(success=True, msg="Login successful")

# ================= MAIN =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
