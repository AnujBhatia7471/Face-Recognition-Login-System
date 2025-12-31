const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const result = document.getElementById("result");

/* ================= CAMERA START ================= */
async function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    result.innerText = "❌ Camera API not supported in this browser";
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" },
      audio: false
    });

    video.srcObject = stream;
    video.setAttribute("playsinline", true); // important for mobile
    video.setAttribute("autoplay", true);

    await video.play();
  } catch (err) {
    console.error(err);
    result.innerText =
      "❌ Camera access denied. Please allow camera permission and use HTTPS.";
  }
}

/* Start camera on page load */
window.addEventListener("load", startCamera);

/* ================= CAPTURE ================= */
function capture() {
  const email = document.getElementById("email").value.trim();

  if (!email) {
    result.innerText = "❌ Email required";
    return;
  }

  if (video.readyState !== 4) {
    result.innerText = "❌ Camera not ready";
    return;
  }

  const ctx = canvas.getContext("2d");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  canvas.toBlob(
    blob => {
      const formData = new FormData();
      formData.append("email", email);
      formData.append("image", blob, "capture.jpg");

      fetch("/login", {
        method: "POST",
        body: formData
      })
        .then(res => res.json())
        .then(data => {
          if (data.success) {
            result.innerHTML = `
              ✅ Login successful<br>
              Email: ${data.email}<br>
              Similarity: ${data.similarity}
            `;
          } else {
            result.innerText = "❌ " + data.msg;
          }
        })
        .catch(() => {
          result.innerText = "❌ Server error";
        });
    },
    "image/jpeg",
    0.95
  );
}
