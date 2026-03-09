const imageInput = document.getElementById("imageInput");
const previewImage = document.getElementById("previewImage");
const resultDiv = document.getElementById("result");


// -----------------------------
// Preview Selected Image
// -----------------------------
imageInput.addEventListener("change", function () {
    const file = this.files[0];

    if (!file) {
        previewImage.src = "";
        return;
    }

    const reader = new FileReader();
    reader.onload = function (e) {
        previewImage.src = e.target.result;
    };
    reader.readAsDataURL(file);
});


// -----------------------------
// Upload & Analyze Image
// -----------------------------
async function uploadImage() {

    const file = imageInput.files[0];

    if (!file) {
        alert("📤 Please upload an image first.");
        return;
    }

    const formData = new FormData();
    formData.append("image", file);

    resultDiv.innerHTML = "🔍 Analyzing banana... Please wait 🍌";

    try {

        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error("Server error.");
        }

        const data = await response.json();

        if (data.error) {
            resultDiv.innerHTML = "❌ Error: " + data.error;
            return;
        }

        // 🔥 Replace preview image with annotated image
        if (data.image_url) {
            previewImage.src = data.image_url + "?t=" + new Date().getTime();
        }

        // Show analysis message
        resultDiv.innerHTML = `<div class="analysis-text">${data.message}</div>`;

    } catch (error) {
        console.error(error);
        resultDiv.innerHTML = "❌ Something went wrong during analysis.";
    }
}