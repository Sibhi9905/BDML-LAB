# =====================================
# app.py - MailGuard Flask REST API
# =====================================

from flask import Flask, request, jsonify
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

app = Flask(__name__)

# -------------------------------
# Load model and tokenizer once
# -------------------------------
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Load model structure
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )
    
    # Load trained weights
    model.load_state_dict(torch.load("models/distilbert_spam_model.pth", map_location=device))
    model.to(device)
    model.eval()
    
    return tokenizer, model, device

tokenizer, model, device = load_model()

# -------------------------------
# Prediction function
# -------------------------------
def predict(email_text, tokenizer, model, device):
    inputs = tokenizer(
        email_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )
    
    # Move inputs to the correct device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred].item()
    
    label = "🚫 SPAM" if pred == 1 else "📩 NOT SPAM"
    return {"label": label, "confidence": round(confidence*100, 2)}

# -------------------------------
# Routes
# -------------------------------
@app.route('/')
def home():
    return jsonify({"message": "MailGuard API (DistilBERT) is running successfully!"})

@app.route('/predict', methods=['POST'])
def classify():
    try:
        data = request.get_json()
        if not data or 'email' not in data:
            return jsonify({"error": "Missing 'email' field in request."}), 400
        
        email_text = data['email']
        prediction = predict(email_text, tokenizer, model, device)
        
        return jsonify({
            "status": "success",
            "prediction": prediction,
            "input": email_text
        })
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# -------------------------------
# Run Flask app
# -------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
