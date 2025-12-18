from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import torch
import os
import joblib

from data_utils.database import database

class JobClassifierTrainer:
    model_name = "EuroBERT/EuroBERT-210m"
    num_labels = 7  # 7 kategori için
    
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.label_encoder = None
    
    def initialize_model(self):
        """Model ve tokenizer'ı sadece ihtiyaç duyulduğunda yükle"""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if self.model is None:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                trust_remote_code=True,
            )

    def load_model(self, model_dir="./fine_tuned_eurobert"):
        """Önceden kaydedilmiş model ve tokenizer'ı yükle"""
        if os.path.exists(model_dir):
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            if os.path.exists(f"{model_dir}/label_encoder.joblib"):
                self.label_encoder = joblib.load(f"{model_dir}/label_encoder.joblib")
            print(f"Model loaded from {model_dir}")
        else:
            print(f"No saved model found at {model_dir}, initializing from scratch")
            self.initialize_model()
    
    @staticmethod
    def compute_metrics(p):
        preds = p.predictions.argmax(-1)
        return {"accuracy": accuracy_score(p.label_ids, preds)}
    
    def tokenize_function(self, examples, text_column="description"):
        """Tokenization fonksiyonu"""
        if self.tokenizer is None:
            self.initialize_model()
        return self.tokenizer(examples[text_column], padding="max_length", truncation=True)
    
    def prepare_datasets_from_tuples(self, tuple_list, 
                                   description_index=4,
                                   label_index=3,
                                   test_size=0.2,
                                   seed=42):
        """Tuple listesinden dataset hazırla"""
        descriptions = []
        labels = []
        
        for tup in tuple_list:
            if len(tup) > max(description_index, label_index):
                desc = tup[description_index]
                label = tup[label_index]
                
                if desc is None:
                    for idx in range(len(tup)):
                        if idx != description_index and isinstance(tup[idx], str) and len(tup[idx]) > 10:
                            desc = tup[idx]
                            break
                
                if desc and label:
                    descriptions.append(str(desc))
                    labels.append(str(label))
        
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        dataset_dict = {
            "description": descriptions,
            "label_text": labels
        }
        
        dataset = Dataset.from_dict(dataset_dict)
        if self.tokenizer is None:
            self.initialize_model()
        dataset = dataset.map(lambda x: self.tokenize_function(x, "description"), batched=True)
        dataset = dataset.map(lambda x: {"label": self.label_encoder.transform([x["label_text"]])[0]}, batched=False)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        
        split_dataset = dataset.train_test_split(test_size=test_size, seed=seed)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
        
        return train_dataset, eval_dataset
    
    def train(self, train_dataset, eval_dataset, device="cpu"):
        """Modeli eğit"""
        if self.model is None:
            self.initialize_model()
        
        device = torch.device("cpu")
        self.model.to(device)
        for param in self.model.parameters():
            param.data = param.data.cpu()
            if param.grad is not None:
                param.grad.data = param.grad.data.cpu()
        torch.cuda.is_available = lambda: False
        
        training_args = TrainingArguments(
            output_dir='./results',
            eval_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=2,
            num_train_epochs=1,
            weight_decay=0.01,
            gradient_accumulation_steps=16,
            no_cuda=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to="none",
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )
        
        trainer.train()
        results = trainer.evaluate()
        print("Evaluation results:", results)
        
        return trainer
    
    def pipeline(self, tuple_list, device="cpu", description_index=4, label_index=3):
        """Tuple listesiyle tüm pipeline"""
        train_dataset, eval_dataset = self.prepare_datasets_from_tuples(
            tuple_list, 
            description_index=description_index,
            label_index=label_index
        )
        trainer = self.train(train_dataset, eval_dataset, device)
        return trainer
    
    def save_model(self, output_dir="./fine_tuned_eurobert", trainer=None):
        """Modeli kaydet"""
        if trainer is not None:
            trainer.save_model(output_dir)
        elif self.model is not None:
            self.model.save_pretrained(output_dir)
        
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        
        if self.label_encoder is not None:
            joblib.dump(self.label_encoder, f"{output_dir}/label_encoder.joblib")
            print(f"Label encoder saved to {output_dir}/label_encoder.joblib")
        
        print(f"Model saved to {output_dir}")


if __name__ == '__main__':
    tuple_list = database.get_unlabelled_samples(100, 0)
    
    if not tuple_list:
        print("Boş liste!")
        exit(1)
    
    trainer = JobClassifierTrainer()
    trainer.load_model("./fine_tuned_eurobert")  # Önceki modeli yükle
    trained_trainer = trainer.pipeline(tuple_list, description_index=1, label_index=3)
    trainer.save_model("./fine_tuned_eurobert", trained_trainer)

