import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import HF_TOKEN, MODEL_PATH


class InferenceService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if torch.cuda.is_available():
            print(f"✅ GPU доступен! Используется: {torch.cuda.get_device_name(0)}")
        else:
            print("❌ GPU НЕ доступен, работаем на CPU")

        self.tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2",
            token=HF_TOKEN)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "cointegrated/rubert-tiny2",
            token=HF_TOKEN,
            num_labels=2,
        ).to(self.device)

        state_dict = torch.load(str(MODEL_PATH), map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.labels = [
            'Normal', 'Toxic'
        ]

    def predict(self, text: str) -> str:
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = F.softmax(outputs.logits, dim=1)
        pred_id = torch.argmax(outputs.logits, dim=1).item()
        confidence = probs[0][pred_id].item()

        return {"label": self.labels[pred_id],
                "confidence": confidence}