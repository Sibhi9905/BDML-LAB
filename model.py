# =====================================
# model.py - MailGuard DistilBERT Model
# =====================================

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import os
import random
import numpy as np

# --------------------------
# CONFIGURATION
# --------------------------
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "distilbert_spam_model.pth")
DATA_PATH = os.path.join("dataset", "SMSSpamCollection")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_LABELS = 2
BATCH_SIZE = 8
EPOCHS = 5
LR = 2e-5
MAX_LEN = 256
SEED = 42

# --------------------------
# REPRODUCIBILITY
# --------------------------
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --------------------------
# DATASET CLASS
# --------------------------
class SpamDataset(Dataset):
    """Custom dataset for SMS Spam classification using DistilBERT"""
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# --------------------------
# TRAIN FUNCTION
# --------------------------
def train_model():
    print("📥 Loading dataset...")
    data = pd.read_csv(DATA_PATH, sep="\t", header=None, names=["label", "message"])
    data["label"] = data["label"].map({"ham": 0, "spam": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        data["message"], data["label"], test_size=0.2, random_state=SEED
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    train_dataset = SpamDataset(X_train.tolist(), y_train.tolist(), tokenizer)
    test_dataset = SpamDataset(X_test.tolist(), y_test.tolist(), tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    print("⚙️ Initializing DistilBERT model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=NUM_LABELS
    ).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR)

    # --------------------------
    # TRAINING LOOP
    # --------------------------
    print("🚀 Training started...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1} completed | Avg Loss: {avg_loss:.4f}")

    # --------------------------
    # EVALUATION LOOP
    # --------------------------
    print("\n📊 Evaluating model...")
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"📈 Test Accuracy: {accuracy:.4f}")

    # --------------------------
    # SAVE MODEL
    # --------------------------
    print("\n💾 Saving model...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ Model saved successfully at: {MODEL_PATH}")


# --------------------------
# LOAD FUNCTION (for inference)
# --------------------------
def load_model():
    print("🔄 Loading trained DistilBERT model...")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=NUM_LABELS
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("✅ Model loaded and ready for inference.")
    return tokenizer, model


# --------------------------
# PREDICT FUNCTION
# --------------------------
def predict(text, tokenizer, model):
    """Make prediction using loaded model."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LEN).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
        label = "Spam" if pred == 1 else "Not Spam"
    return label


# --------------------------
# MAIN (TRAIN WHEN RUN DIRECTLY)
# --------------------------
if __name__ == "__main__":
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    train_model()
