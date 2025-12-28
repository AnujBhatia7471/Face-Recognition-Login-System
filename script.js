const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const result = document.getElementById("result");

navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => video.srcObject = stream)
  .catch(err => result.innerText = "Camera access denied");

function capture() {
  const email = document.getElementById("email").value;
  if (!email) {
    result.innerText = "❌ Email required";
    return;
  }

  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  canvas.toBlob(blob => {
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
    });
  }, "image/jpeg");
}
