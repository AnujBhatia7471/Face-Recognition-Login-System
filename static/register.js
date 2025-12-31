const canvas = document.getElementById("canvas");
const result = document.getElementById("result");
const sampleBtn = document.getElementById("sampleBtn");
const startBtn = document.getElementById("startBtn");

let sampleCount = 0;
const MAX_SAMPLES = 5;

async function startRegistration() {
  const emailVal = email.value.trim();
  const passVal = password.value;

  if (!emailVal || !passVal) {
    result.innerText = "Email and password required";
    return;
  }

  await startCamera();
  result.innerText = "Camera started. Take sample 1";
  sampleBtn.disabled = false;
  startBtn.disabled = true;
}

function takeSample() {
  if (sampleCount >= MAX_SAMPLES) return;

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);

  canvas.toBlob(blob => {
    const form = new FormData();
    form.append("email", email.value.trim());
    form.append("password", password.value);
    form.append("image", blob);

    fetch("/register", { method: "POST", body: form })
      .then(res => res.json())
      .then(data => {
        result.innerText = data.msg;

        if (!data.success) {
          sampleBtn.disabled = true;
          stopCamera();
          return;
        }

        sampleCount++;

        if (data.completed) {
          sampleBtn.disabled = true;
          stopCamera();
        } else {
          sampleBtn.innerText = `Take Sample (${sampleCount + 1} / 5)`;
        }
      })
      .catch(() => {
        result.innerText = "Server error";
      });
  }, "image/jpeg", 0.95);
}
