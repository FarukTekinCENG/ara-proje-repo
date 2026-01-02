import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd

from model import ModelPredictor


@dataclass
class PredictionResult:
    description: str
    predicted_class: int
    predicted_label: Optional[str]
    confidence: float
    probs: List[float]


class DemoFeatures:
    def __init__(
        self,
        model_dir: str = "./trained_models/model1",
        fallback_model: str = "EuroBERT/EuroBERT-210m",
        num_labels: int = 7,
        label_list: Optional[Sequence[str]] = None,
    ):
        self.model_dir = model_dir
        self.fallback_model = fallback_model
        self.num_labels = num_labels
        self._label_list_override = list(label_list) if label_list is not None else None

        self.predictor = ModelPredictor(
            model_path=self.model_dir,
            fallback_model=self.fallback_model,
            num_labels=self.num_labels,
        )

        self._label_list: Optional[List[str]] = None

    def load_model(self) -> None:
        self.predictor.load_model()
        self._label_list = self._load_label_list()

    def _load_label_list(self) -> Optional[List[str]]:
        if self._label_list_override is not None:
            return list(self._label_list_override)

        # Try to load label encoder from the saved model folder.
        # This is the most reliable mapping for UI display.
        le_path = os.path.join(self.model_dir, "label_encoder.joblib")
        if os.path.exists(le_path):
            try:
                import joblib

                le = joblib.load(le_path)
                classes = getattr(le, "classes_", None)
                if classes is not None:
                    return [str(x) for x in list(classes)]
            except Exception:
                pass

        # Fallback: use the training code's fixed label list.
        try:
            from train import JobClassifierTrainer

            return list(getattr(JobClassifierTrainer, "ALL_LABELS", [])) or None
        except Exception:
            return None

    def _class_to_label(self, class_id: int) -> Optional[str]:
        if self._label_list is None:
            self._label_list = self._load_label_list()
        if not self._label_list:
            return None
        if 0 <= int(class_id) < len(self._label_list):
            return str(self._label_list[int(class_id)])
        return None

    def predict_one(self, description: str) -> PredictionResult:
        if self.predictor.model is None or self.predictor.tokenizer is None:
            self.load_model()

        out = self.predictor.predict(description)
        predicted_class = int(out["predicted_class"])
        return PredictionResult(
            description=str(description),
            predicted_class=predicted_class,
            predicted_label=self._class_to_label(predicted_class),
            confidence=float(out["confidence"]),
            probs=list(out["probs"]),
        )

    def predict_many(
        self,
        descriptions: Union[Sequence[str], pd.Series],
    ) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for text in list(descriptions):
            r = self.predict_one(str(text))
            rows.append(
                {
                    "description": r.description,
                    "predicted_class": r.predicted_class,
                    "predicted_label": r.predicted_label,
                    "confidence": r.confidence,
                    "probs": r.probs,
                }
            )
        return pd.DataFrame(rows)

    def load_test_df(
        self,
        csv_path: str,
        description_col: str = "description",
        label_col: str = "label",
        encoding: str = "utf-8",
    ) -> pd.DataFrame:
        df = pd.read_csv(csv_path, encoding=encoding)
        if description_col not in df.columns:
            raise KeyError(f"CSV must contain description column '{description_col}'")
        if label_col not in df.columns:
            raise KeyError(f"CSV must contain label column '{label_col}'")
        return df[[description_col, label_col]].copy()

    def evaluate(
        self,
        test_df: pd.DataFrame,
        description_col: str = "description",
        label_col: str = "label",
    ) -> Tuple[pd.DataFrame, float]:
        if description_col not in test_df.columns:
            raise KeyError(f"test_df must contain '{description_col}'")
        if label_col not in test_df.columns:
            raise KeyError(f"test_df must contain '{label_col}'")

        preds_df = self.predict_many(test_df[description_col])
        out_df = test_df.copy()
        out_df = out_df.rename(columns={description_col: "description", label_col: "true_label"})

        out_df["predicted_class"] = preds_df["predicted_class"].astype(int)
        out_df["predicted_label"] = preds_df["predicted_label"]
        out_df["confidence"] = preds_df["confidence"].astype(float)

        # Correctness: if true labels are strings, compare to predicted_label.
        # If they are numeric, compare to predicted_class.
        def _is_intish(x: Any) -> bool:
            try:
                if x is None:
                    return False
                s = str(x).strip()
                if not s:
                    return False
                f = float(s)
                return f.is_integer()
            except Exception:
                return False

        true_is_numeric = out_df["true_label"].map(_is_intish).all()
        if true_is_numeric:
            out_df["true_label_int"] = out_df["true_label"].map(lambda x: int(float(x)))
            out_df["is_correct"] = out_df["true_label_int"].astype(int) == out_df["predicted_class"].astype(int)
        else:
            out_df["is_correct"] = out_df["true_label"].astype(str) == out_df["predicted_label"].astype(str)

        accuracy = float(out_df["is_correct"].mean()) if len(out_df) else 0.0
        return out_df, accuracy
