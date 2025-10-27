document.getElementById("predictBtn").addEventListener("click", async () => {
    const emailText = document.getElementById("emailInput").value.trim();
    if (!emailText) {
        alert("Please enter email content first!");
        return;
    }

    const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: emailText })
    });

    const data = await response.json();

    const resultBox = document.getElementById("resultBox");
    document.getElementById("predictionLabel").textContent = `Prediction: ${data.label}`;
    document.getElementById("predictionConfidence").textContent = `Confidence: ${data.confidence}%`;
    resultBox.classList.remove("hidden");
});
