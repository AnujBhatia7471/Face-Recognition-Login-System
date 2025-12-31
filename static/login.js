const canvas = document.getElementById("canvas");
const result = document.getElementById("result");

/* ================= PASSWORD LOGIN ================= */
function passwordLogin() {
  fetch("/login/password", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      email: p_email.value.trim(),
      password: p_password.value
    })
  })
  .then(r => r.json())
  .then(d => {
    result.innerText = d.msg;

    if (d.success) {
      // 🔥 MANUAL REDIRECT (THIS WAS MISSING)
      window.location.href = "/dashboard";
    }
  })
  .catch(() => {
    result.innerText = "Server error";
  });
}

/* ================= FACE LOGIN ================= */
async function faceLogin() {
  const emailVal = f_email.value.trim();
  if (!emailVal) {
    result.innerText = "Email required";
    return;
  }

  result.innerText = "Starting camera...";
  await startCamera();

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);

  canvas.toBlob(blob => {
    const f = new FormData();
    f.append("email", emailVal);
    f.append("image", blob);

    fetch("/login/face", { method: "POST", body: f })
      .then(r => r.json())
      .then(d => {
        result.innerText = d.msg;

        if (d.success) {
          stopCamera();
          // 🔥 MANUAL REDIRECT (THIS WAS MISSING)
          window.location.href = "/dashboard";
        }
      })
      .catch(() => {
        result.innerText = "Server error";
        stopCamera();
      });
  }, "image/jpeg", 0.95);
}
