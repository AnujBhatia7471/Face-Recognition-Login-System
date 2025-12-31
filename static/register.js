const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const result = document.getElementById("result");

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;
  await video.play();
}
window.onload = startCamera;

function register() {
  const email = document.getElementById("email").value;
  if (!email) {
    result.className = "error";
    result.innerText = "❌ Email required";
    return;
  }

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);

  canvas.toBlob(blob => {
    const formData = new FormData();
    formData.append("email", email);
    formData.append("image", blob);

    fetch("/register", { method: "POST", body: formData })
      .then(res => res.json())
      .then(data => {
        result.className = data.success ? "success" : "error";
        result.innerText = data.success
          ? "✅ Registration Successful"
          : "❌ " + data.msg;
      });
  });
}
