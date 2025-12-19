import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


class ModelPredictor:
    def __init__(
        self,
        model_path="./fine_tuned_eurobert",
        fallback_model="EuroBERT/EuroBERT-210m",
        num_labels=7,
    ):
        self.model_path = model_path
        self.fallback_model = fallback_model
        self.num_labels = num_labels

        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        # 1️⃣ Yerel model VARSA
        if os.path.isdir(self.model_path):
            print(f"Local model found: {self.model_path}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )

        # 2️⃣ Yerel model YOKSA → base modele düş
        else:
            print(
                f"WARNING: Local model not found ({self.model_path}). "
                f"Falling back to base model: {self.fallback_model}"
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.fallback_model,
                trust_remote_code=True,
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.fallback_model,
                num_labels=self.num_labels,
                trust_remote_code=True,
            )

        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str):
        if self.model is None or self.tokenizer is None:
            self.load_model()

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        confidence, predicted_class = torch.max(probs, dim=-1)

        return {
            "predicted_class": int(predicted_class.item()),
            "confidence": float(confidence.item()),
        }

