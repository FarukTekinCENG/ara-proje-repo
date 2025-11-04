from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

class train:
    model_name = "EuroBERT/EuroBERT-610m"
    num_labels = 5
    train_dataset=[]
    eval_dataset=[]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        trust_remote_code=True,
        device_map="auto",
    )
        
    @staticmethod
    def compute_metrics(p):
        preds = p.predictions.argmax(-1)
        return {"accuracy": accuracy_score(p.label_ids, preds)}

    @staticmethod
    def tokenize_function(examples, description_col_name="Job Description"):
        return train.tokenizer(examples[description_col_name], padding="max_length", truncation=True)

    @staticmethod
    def loadDataset(dataset_path="job_descriptions.csv", nrows=5000,
     description_col_name = "Job Description"):
        df = pd.read_csv(dataset_path, nrows=nrows)

        # Label encoding
        le = LabelEncoder()
        df["Work Type"] = le.fit_transform(df["Work Type"])
        print("Label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

        dataset = Dataset.from_pandas(df)
        
        dataset = dataset.map(lambda x: train.tokenize_function(x, description_col_name), batched=True)

        dataset = dataset.rename_column("Work Type", "label")
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

        split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
        train.train_dataset = split_dataset["train"]
        train.eval_dataset = split_dataset["test"]

    @staticmethod
    def pipeline():
        train.loadDataset()

        # GPU cihaz kontrolü
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)
        train.model.to(device)

        training_args = TrainingArguments(
            output_dir='./results',
            eval_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=4,
            num_train_epochs=3,
            weight_decay=0.01,
            gradient_accumulation_steps=8,
            fp16=True,
        )
        trainer = Trainer(
            model=train.model,
            args=training_args,
            train_dataset=train.train_dataset,
            eval_dataset=train.eval_dataset,
            tokenizer=train.tokenizer,
            compute_metrics=train.compute_metrics
        )

        trainer.train()

        # test result
        results = trainer.evaluate()
        print(results)

        # Trainer ile model kaydetme
        # trainer.save_model("./fine_tuned_eurobert")
        # train.tokenizer.save_pretrained("./fine_tuned_eurobert")

if __name__ == '__main__':
    train.pipeline()
