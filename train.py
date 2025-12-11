from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import torch

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
                                   description_index=4,  # None gördüğümüz index
                                   label_index=3,        # 'Full-time' gördüğümüz index
                                   test_size=0.2,
                                   seed=42):
        """
        Tuple listesinden dataset hazırla
        
        Args:
            tuple_list: Tuple listesi
            description_index: Açıklama metninin index'i
            label_index: Etiketin index'i
            test_size: Eval split oranı
            seed: Random seed
            
        Returns:
            train_dataset, eval_dataset
        """
        if not tuple_list or not isinstance(tuple_list, list):
            raise ValueError("Geçerli bir tuple listesi sağlanmalı")
        
        print(f"Tuple uzunluğu: {len(tuple_list[0])}")
        print(f"Örnek tuple: {tuple_list[0]}")
        
        # # Tuple yapısını analiz et
        # for i, tup in enumerate(tuple_list[:3]):
        #     print(f"Tuple {i}: {tup}")
        #     for j, item in enumerate(tup):
        #         print(f"  [{j}]: {type(item)} = {item}")
        
        # Description ve label'ları ayıkla
        descriptions = []
        labels = []
        
        for tup in tuple_list:
            if len(tup) > max(description_index, label_index):
                desc = tup[description_index]
                label = tup[label_index]
                
                # Eğer description None ise, başka bir field kullan
                if desc is None:
                    # Farklı index'leri dene
                    for idx in range(len(tup)):
                        if idx != description_index and isinstance(tup[idx], str) and len(tup[idx]) > 10:
                            desc = tup[idx]
                            print(f"None description için alternatif bulundu: index {idx}")
                            break
                
                if desc and label:
                    descriptions.append(str(desc))
                    labels.append(str(label))
        
        print(f"\nToplam {len(descriptions)} geçerli örnek bulundu")
        print(f"İlk description: {descriptions[0][:100]}..." if descriptions else "No descriptions")
        print(f"İlk label: {labels[0]}" if labels else "No labels")
        
        # Label encoding
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(labels)
        print("\nLabel mapping:", dict(zip(self.label_encoder.classes_, 
                                          self.label_encoder.transform(self.label_encoder.classes_))))
        
        # Veriyi Dataset formatına çevir
        dataset_dict = {
            "description": descriptions,
            "label_text": labels  # Orijinal label'ları da sakla
        }
        
        dataset = Dataset.from_dict(dataset_dict)
        
        # Tokenizer'ı başlat
        if self.tokenizer is None:
            self.initialize_model()
        
        # Tokenization
        dataset = dataset.map(
            lambda x: self.tokenize_function(x, "description"), 
            batched=True
        )
        
        # Label encoding uygula
        dataset = dataset.map(
            lambda x: {"label": self.label_encoder.transform([x["label_text"]])[0]},
            batched=False
        )
        
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        
        # Split
        split_dataset = dataset.train_test_split(test_size=test_size, seed=seed)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
        
        print(f"\nTrain set: {len(train_dataset)} örnek")
        print(f"Eval set: {len(eval_dataset)} örnek")
        
        return train_dataset, eval_dataset
    
    # train() metodunu güncelleyelim
    def train(self, train_dataset, eval_dataset, device="cpu"):
        """
        Modeli eğit
        
        Args:
            train_dataset: Eğitim dataset'i
            eval_dataset: Eval dataset'i
            device: "cpu" veya "cuda"
                
        Returns:
            trainer: Eğitilmiş trainer objesi
        """
        # Modeli başlat
        self.initialize_model()
        
        # KESİNLİKLE SADECE CPU KULLAN
        device = torch.device("cpu")
        print(f"Using device: {device}")
        self.model.to(device)
        
        # Modelin tüm parametrelerini CPU'ya gönder
        for param in self.model.parameters():
            param.data = param.data.cpu()
            if param.grad is not None:
                param.grad.data = param.grad.data.cpu()
        
        # CUDA'yı tamamen devre dışı bırak
        torch.cuda.is_available = lambda: False
        
        # Training arguments - ÇOK KÜÇÜK BATCH SIZE
        training_args = TrainingArguments(
            output_dir='./results',
            eval_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=1,    # ÇOK KÜÇÜK
            per_device_eval_batch_size=2,     # ÇOK KÜÇÜK  
            num_train_epochs=1,               # SADECE 1 EPOCH
            weight_decay=0.01,
            gradient_accumulation_steps=16,   # ÇOK YÜKSEK
            no_cuda=True,                     # KESİNLİKLE NO CUDA
            dataloader_pin_memory=False,      # MEMORY PIN'leme yok
            remove_unused_columns=False,      # Gerekli olabilir
            report_to="none",                 # WandB vs. yok
        )
        
        # Trainer - CUDA kullanma
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )
        
        # Eğitim
        trainer.train()
        
        # Değerlendirme
        results = trainer.evaluate()
        print("Evaluation results:", results)
        
        return trainer

    def pipeline(self, tuple_list, device="cpu", description_index=4, label_index=3):
        """
        Tuple listesiyle tüm pipeline
        
        Args:
            tuple_list: Tuple listesi
            device: "cpu" veya "cuda"
            description_index: Description index'i
            label_index: Label index'i
            
        Returns:
            trainer: Eğitilmiş trainer objesi
        """
        print("Starting pipeline...")
        
        # 1. Dataset'leri hazırla
        print("Preparing datasets...")
        train_dataset, eval_dataset = self.prepare_datasets_from_tuples(
            tuple_list, 
            description_index=description_index,
            label_index=label_index
        )
        
        # 2. Modeli eğit
        print("Training model...")
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
        
        # Label encoder'ı da kaydet
        if self.label_encoder is not None:
            import joblib
            joblib.dump(self.label_encoder, f"{output_dir}/label_encoder.joblib")
            print(f"Label encoder saved to {output_dir}/label_encoder.joblib")
        
        print(f"Model saved to {output_dir}")

# train.py'nin son kısmını güncelleyelim
if __name__ == '__main__':
    # 1. Veriyi tuple listesi olarak yükle
    tuple_list = database.get_unlabelled_samples(100, 0)
    
    print(f"Veri tipi: {type(tuple_list)}")
    print(f"Uzunluk: {len(tuple_list)}")
    
    if not tuple_list:
        print("Boş liste!")
        exit(1)
    
    # 2. İlk tuple'ı analiz et - DOĞRU INDEX'LERİ BUL
    first_tuple = tuple_list[0]
    print(f"\nİlk tuple: {first_tuple}")
    print(f"Tuple uzunluğu: {len(first_tuple)}")
    
    print("\nTuple yapısı:")
    for i, item in enumerate(first_tuple):
        print(f"[{i}]: {type(item).__name__} = {repr(str(item)[:100])}")
    
    # 3. DOĞRU INDEX'LER:
    # index 0: id (int)
    # index 1: description (uzun string) 
    # index 2: is_labelled ('FALSE' string'i) - BU ETİKET DEĞİL!
    # index 3: label ('Full-time' string'i) - BU GERÇEK ETİKET!
    # index 4: None
    # index 5: None
    
    # 4. Trainer oluştur - DOĞRU INDEX'LERLE
    trainer = JobClassifierTrainer()
    
    try:
        # Pipeline'ı çalıştır - description_index=1, label_index=3
        print("\n=== DOĞRU INDEX'LERLE BAŞLIYOR ===")
        print("description_index = 1 (uzun açıklama metni)")
        print("label_index = 3 ('Full-time', 'Part-time' vs.)")
        
        trained_trainer = trainer.pipeline(
            tuple_list, 
            device="cpu",  # SADECE CPU KULLAN
            description_index=1,  # Doğru: description burada
            label_index=3         # Doğru: label burada
        )
        
        # Modeli kaydet
        trainer.save_model("./fine_tuned_eurobert", trained_trainer)
        
    except Exception as e:
        print(f"\nHata: {e}")
        print("\nSorun çözümleri:")
        print("1. Batch size'ı daha da küçült:")
        print("   per_device_train_batch_size=2, per_device_eval_batch_size=4")
        print("\n2. Gradient accumulation steps'ı artır:")
        print("   gradient_accumulation_steps=8")
        print("\n3. Daha az epoch:")
        print("   num_train_epochs=1 veya 2")
