# Gerekli importlar
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from typing import Dict, Union, List

class ModelPredictor:
    def __init__(self, model_path: str = "./fine_tuned_eurobert"):
        """
        Model yükleyici sınıfı
        
        Args:
            model_path: Fine-tune edilmiş modelin yolu
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.label_mapping = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self):
        """Model ve tokenizer'ı yükler"""
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Modeli uygun cihaza gönder
        self.model.to(self.device)
        self.model.eval()  # Evaluation moduna al
        
        # 7 KATEGORİLİ ETİKET MAPPING
        self.label_mapping = {
            0: "Contract",
            1: "Full-time", 
            2: "Internship",
            3: "Other",
            4: "Part-time",
            5: "Temporary",
            6: "Volunteer"
        }
        
        print(f"Model loaded on {self.device}")
        
    def predict(self, text: Union[str, List[str]]) -> Dict:
        """
        Metin(ler) için tahmin yapar
        
        Args:
            text: Tahmin yapılacak metin veya metin listesi
            
        Returns:
            Tahmin sonuçları
        """
        if self.model is None or self.tokenizer is None:
            self.load_model()
            
        # Tokenization
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        )
        
        # Input'ları aynı cihaza gönder
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Tahmin
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
        
        # CPU'ya geri getir
        predictions = predictions.cpu().numpy()
        probabilities = probabilities.cpu().numpy()
        
        # Sonuçları formatla
        results = []
        for i, pred in enumerate(predictions):
            result = {
                "text": text[i] if isinstance(text, list) else text,
                "predicted_label": int(pred),
                "predicted_class": self.label_mapping.get(int(pred), "Unknown"),
                "confidence": float(probabilities[i][pred]),
                "probabilities": {
                    self.label_mapping.get(j, f"Label_{j}"): float(prob)
                    for j, prob in enumerate(probabilities[i])
                }
            }
            
            results.append(result)
        
        return results if isinstance(text, list) else results[0]
    
    def predict_batch(self, texts: List[str], batch_size: int = 8) -> List[Dict]:
        """
        Büyük batch'ler için tahmin yapar
        
        Args:
            texts: Tahmin yapılacak metin listesi
            batch_size: Batch boyutu
            
        Returns:
            Tahmin sonuçları listesi
        """
        all_results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_results = self.predict(batch)
            all_results.extend(batch_results)
            
        return all_results


# Kullanım örneği
if __name__ == "__main__":
    # Modeli oluştur
    predictor = ModelPredictor("./fine_tuned_eurobert")
    
    # Test metinleri
    test_texts = [
        "We are looking for a full-time software engineer with 5 years experience in Java and Spring Framework.",
        "Part-time marketing assistant needed for 20 hours per week, remote work possible.",
        "Summer internship program for computer science students - 3 months duration.",
        "Contract position for senior DevOps engineer, 6-month project with possible extension.",
        "Temporary administrative assistant needed for maternity leave coverage.",
        "Volunteer opportunity at local non-profit organization, teaching coding to kids.",
        "We need a project manager for a new initiative, employment type flexible.",
        "Senior data scientist position with competitive salary and benefits package."
    ]
    
    print("=" * 60)
    print("7 KATEGORİLİ İŞ TİPİ TAHMİN MODELİ")
    print("=" * 60)
    
    # Tek bir metin için tahmin
    print("\n1. TEK METİN TAHMİNİ:")
    print("-" * 40)
    single_result = predictor.predict(test_texts[0])
    
    print(f"Metin: {single_result['text']}")
    print(f"\nTahmin: {single_result['predicted_class']}")
    print(f"Güven: {single_result['confidence']:.2%}")
    print(f"Label: {single_result['predicted_label']}")
    
    print("\nTüm olasılıklar:")
    for class_name, prob in single_result['probabilities'].items():
        print(f"  {class_name}: {prob:.2%}")
    
    # Batch tahmin
    print("\n\n2. BATCH TAHMİN SONUÇLARI:")
    print("-" * 40)
    batch_results = predictor.predict_batch(test_texts)
    
    for i, res in enumerate(batch_results, 1):
        print(f"{i}. {res['text'][:60]}...")
        print(f"   → {res['predicted_class']} (Güven: {res['confidence']:.2%})")
    
    print("\n" + "=" * 60)
    print("MODEL HAZIR - Toplam 7 kategori:")
    for label, class_name in predictor.label_mapping.items():
        print(f"  {label}: {class_name}")
    print("=" * 60)
