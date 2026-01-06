import torch
import os
import joblib
import numpy as np
import inspect
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score
from collections import Counter
from torch.nn import CrossEntropyLoss
from transformers import Trainer

# pytorch optimizasyonları
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

class JobClassifierTrainer:
    #model_name="EuroBERT/EuroBERT-210m",
    model_name = "distilbert-base-uncased"

    # 🔒 SABİT LABEL UZAYI (ASLA DEĞİŞMEZ)
    ALL_LABELS = [
        "Contract",
        "Full-time",
        "Internship",
        "Other",
        "Part-time",
        "Temporary",
        "Volunteer",
    ]

    num_labels = len(ALL_LABELS)

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.label_encoder = None
        self.trainer = None

    # --------------------------------------------------
    # MODEL + TOKENIZER + LABEL ENCODER INIT
    # --------------------------------------------------
    def initialize_model(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                fix_mistral_regex=True
            )

        # if self.model is None:
        #     self.model = AutoModelForSequenceClassification.from_pretrained(
        #         self.model_name,
        #         num_labels=self.num_labels,
        #         trust_remote_code=True
        #     )
        if self.model is None:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                trust_remote_code=True,
                ignore_mismatched_sizes=True
            )

            # 🔴 SEÇENEK A: classification head'i bilinçli şekilde sıfırla
            if hasattr(self.model, "classifier"):
                self.model.classifier.reset_parameters()
                print("⚠️ Base classifier: classification head reset")
            elif hasattr(self.model, "score"):
                self.model.score.reset_parameters()
                print("⚠️ Base classifier: score head reset")
            else:
                raise RuntimeError(
                    "Classification head not found. "
                    "Unexpected model architecture."
                )

        # 🔒 LABEL ENCODER SADECE BURADA FIT EDİLİR
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(self.ALL_LABELS)

    # --------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------
    def load_model(self, model_dir="./base_classifier"):
        if os.path.exists(model_dir):
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_dir,
                trust_remote_code=True,
                fix_mistral_regex=True
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_dir,
                trust_remote_code=True
            )

            le_path = os.path.join(model_dir, "label_encoder.joblib")
            if os.path.exists(le_path):
                self.label_encoder = joblib.load(le_path)
            else:
                # 🔒 ASLA yeniden fit etme
                self.label_encoder = LabelEncoder()
                self.label_encoder.fit(self.ALL_LABELS)

            print(f"Model loaded from {model_dir}")
        else:
            print("Base classifier not found. Creating base classifier...")
            self.initialize_model()
            self.save_model(model_dir)

    # --------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------
    def save_model(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        if self.label_encoder:
            joblib.dump(
                self.label_encoder,
                os.path.join(output_dir, "label_encoder.joblib"),
            )

        print(f"Model saved to {output_dir}")


    # --------------------------------------------------
    # TOKENIZATION
    # --------------------------------------------------
    def tokenize_function(self, examples):
        return self.tokenizer(
            examples["description"],
            truncation=True,
            padding="max_length",
            max_length=256,
        )

    def prepare_datasets_from_tuples(
        self,
        tuple_list,
        description_index=1,
        label_index=3,
        test_size=0.2,
        seed=42,
        split=True,
    ):
        if self.label_encoder is None:
            raise RuntimeError(
                "LabelEncoder not initialized. "
                "It must be fitted once during base model initialization."
            )

        descriptions = []
        labels = []

        skipped_samples = 0
        skipped_label_counter = {}

        valid_label_set = set(self.label_encoder.classes_)

        for row in tuple_list:
            if len(row) <= max(description_index, label_index):
                continue

            desc = row[description_index]
            label = row[label_index]

            if not desc or not label:
                skipped_samples += 1
                continue

            label = str(label)

            # ❌ BİREBİR EŞLEŞME YOKSA AT
            if label not in valid_label_set:
                skipped_samples += 1
                skipped_label_counter[label] = (
                    skipped_label_counter.get(label, 0) + 1
                )
                continue

            descriptions.append(str(desc))
            labels.append(label)

        # 🚨 UYARI SİNYALİ
        if skipped_samples > 0:
            print(
                f"⚠️ WARNING: {skipped_samples} samples skipped due to unknown labels."
            )
            for lbl, cnt in skipped_label_counter.items():
                print(f"   - '{lbl}': {cnt} samples ignored")

        if not descriptions:
            raise ValueError(
                "No valid samples left after label filtering. "
                "Check label consistency."
            )

        # ✅ SADECE GÜVENLİ TRANSFORM
        encoded_labels = self.label_encoder.transform(labels)

        print(f"Label encoder classes (fixed): {self.label_encoder.classes_}")
        print(f"Number of samples used for training: {len(descriptions)}")

        dataset = Dataset.from_dict(
            {
                "description": descriptions,
                "label": encoded_labels,
            }
        )

        dataset = dataset.map(self.tokenize_function, batched=True)
        dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "label"],
        )

        if split:
            split_ds = dataset.train_test_split(
                test_size=test_size, seed=seed
            )
            return split_ds["train"], split_ds["test"]

        return dataset, None


    # --------------------------------------------------
    # TRAIN
    # --------------------------------------------------
    def train(self, train_dataset, eval_dataset, num_train_epochs=5, learning_rate=2e-5):
        if self.model is None:
            self.initialize_model()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f">> Training on {device}")
        self.model.to(device)

        fp16 = torch.cuda.is_available() and not torch.cuda.is_bf16_supported()
        bf16 = torch.cuda.is_bf16_supported()

        # ⚠️ transformers sürüm farkları nedeniyle TrainingArguments parametrelerini filtrele
        sig = inspect.signature(TrainingArguments.__init__)
        allowed = set(sig.parameters.keys())

        args_kwargs = {
            "output_dir": "./results",
            "per_device_train_batch_size": 16,
            "per_device_eval_batch_size": 16,
            "gradient_accumulation_steps": 1,
            "num_train_epochs": num_train_epochs,                         # 1 AZ > 3-5 OLMALI
            "learning_rate": learning_rate,
            "weight_decay": 0.01,
            "logging_steps": 50,
            "overwrite_output_dir": True,
            "remove_unused_columns": False,
            "report_to": "none",
            "fp16": fp16,
            "bf16": bf16,
        }

        # Checkpoint/disk bloat önleme (destekleniyorsa)
        if "save_strategy" in allowed:
            args_kwargs["save_strategy"] = "no"
        if "save_total_limit" in allowed:
            args_kwargs["save_total_limit"] = 1
        if "evaluation_strategy" in allowed:
            args_kwargs["evaluation_strategy"] = "no"
        elif "eval_strategy" in allowed:
            args_kwargs["eval_strategy"] = "no"

        filtered_kwargs = {k: v for k, v in args_kwargs.items() if k in allowed}
        training_args = TrainingArguments(**filtered_kwargs)

        # trainer = Trainer(
        #     model=self.model,
        #     args=training_args,
        #     train_dataset=train_dataset,
        #     #eval_dataset=eval_dataset,  # None olabilir, sorun yok
        #     tokenizer=self.tokenizer,
        #     #compute_metrics=self.compute_metrics if eval_dataset is not None else None,
        # )
        
        # class weights hesapla
        labels = train_dataset["label"]
        class_weights = JobClassifierTrainer.compute_class_weights_from_labels(
            labels,
            num_labels=self.num_labels
        ).to(self.model.device)

        trainer = WeightedLossTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
            class_weights=class_weights,
        )

        self.trainer = trainer

        #
        trainer.train()
        return trainer

    @staticmethod
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, predictions)
        }
    
    @staticmethod
    def compute_class_weights_from_labels(encoded_labels, num_labels):
        counts = Counter(encoded_labels)
        total = sum(counts.values())

        weights = []
        for i in range(num_labels):
            # hiç yoksa çok büyük ceza
            w = total / counts[i] if i in counts else total
            weights.append(w)

        return torch.tensor(weights, dtype=torch.float)

class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        **kwargs,   # 🔥 kritik
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = CrossEntropyLoss(
            weight=self.class_weights
        ) if self.class_weights is not None else CrossEntropyLoss()

        loss = loss_fct(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )

        return (loss, outputs) if return_outputs else loss
