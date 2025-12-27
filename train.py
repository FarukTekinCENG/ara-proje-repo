from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import torch
import os
import joblib

from data_utils.database import database


class JobClassifierTrainer:
    model_name = "EuroBERT/EuroBERT-210m"
    num_labels = 7

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.label_encoder = None

    def initialize_model(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

        if self.model is None:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                trust_remote_code=True
            )

    def load_model(self, model_dir="./fine_tuned_eurobert"):
        if os.path.exists(model_dir):
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_dir,
                trust_remote_code=True
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_dir,
                trust_remote_code=True
            )

            le_path = os.path.join(model_dir, "label_encoder.joblib")
            if os.path.exists(le_path):
                self.label_encoder = joblib.load(le_path)

            print(f"Model loaded from {model_dir}")
        else:
            print("No saved model found, initializing from base model")
            self.initialize_model()

    @staticmethod
    def compute_metrics(p):
        preds = p.predictions.argmax(-1)
        return {"accuracy": accuracy_score(p.label_ids, preds)}

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
        descriptions = []
        labels = []

        for row in tuple_list:
            if len(row) <= max(description_index, label_index):
                continue

            desc = row[description_index]
            label = row[label_index]

            if desc and label:
                descriptions.append(str(desc))
                labels.append(str(label))

        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(labels)

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
            split_ds = dataset.train_test_split(test_size=test_size, seed=seed)
            return split_ds["train"], split_ds["test"]
        else:
            return dataset, None

    def train(self, train_dataset, eval_dataset):
        if self.model is None:
            self.initialize_model()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f">> Training on {device}")

        self.model.to(device)

        # If no eval dataset provided, disable evaluation strategy to avoid Trainer errors
        eval_strategy = "epoch" if eval_dataset is not None else "no"

        training_args = TrainingArguments(
            output_dir="./results",
            eval_strategy=eval_strategy,
            learning_rate=2e-5,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=8,
            num_train_epochs=1,
            weight_decay=0.01,
            remove_unused_columns=False,
            report_to="none",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()
        return trainer

    def evaluate(self, test_dataset):
        """Evaluate the current model on a provided test dataset.

        The `test_dataset` must be tokenized and formatted similarly to the training dataset.
        Returns the metrics dict from `Trainer.evaluate()`.
        """
        if self.model is None:
            self.initialize_model()

        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )

        return trainer.evaluate(eval_dataset=test_dataset)

    def save_model(self, output_dir="./fine_tuned_eurobert", trainer=None):
        os.makedirs(output_dir, exist_ok=True)

        if trainer:
            trainer.save_model(output_dir)
        else:
            self.model.save_pretrained(output_dir)

        self.tokenizer.save_pretrained(output_dir)

        if self.label_encoder:
            joblib.dump(
                self.label_encoder,
                os.path.join(output_dir, "label_encoder.joblib"),
            )

        print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    samples = database.get_unlabelled_samples(100, 0)

    if not samples:
        print("Boş veri geldi")
        exit(1)

    trainer = JobClassifierTrainer()
    trainer.load_model("./fine_tuned_eurobert")

    # Prepare dataset WITHOUT an automatic train/test split: incoming data should be used only for training.
    train_ds, eval_ds = trainer.prepare_datasets_from_tuples(
        samples,
        description_index=1,
        label_index=3,
        split=False,
    )

    trained_trainer = trainer.train(train_ds, None)
    trainer.save_model("./fine_tuned_eurobert", trained_trainer)

