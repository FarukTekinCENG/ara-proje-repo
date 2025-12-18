import os
import sys
import numpy as np

# Diversity için gerekli kütüphaneler
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from train import JobClassifierTrainer
from model import ModelPredictor
from data_utils.database import database

class ActiveLearning:
    hyper_params = {
        "N": 5,           # Her turda seçilecek örnek sayısı
        "POOL_SIZE": 2000 # Diversity için RAM'e çekilecek aday havuz limiti
    }
    
    # Embedding modelini bir kere yükle
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    @staticmethod
    def model_predict(max_samples=None):
        """
        Uncertainty Sampling için tahmin yapar ve DB'ye skor yazar.
        Modelin kafasının karıştığı örnekleri (confidence düşük) bulmak için.
        """
        predictor = ModelPredictor("./fine_tuned_eurobert")
        batch_size = 1000
        offset = 0
        total_processed = 0
        page = 0

        while True:
            if max_samples is not None and total_processed >= max_samples:
                break
            
            # Etiketlenmemiş (modelin henüz eğitilmediği) verileri getir
            batch = database.get_unlabelled_samples(batch_size, offset)
            
            if not batch:
                break
                
            print(f"Page {page}: {len(batch)} records predicting...")
            total_processed += len(batch)
            page += 1

            for x in batch:
                # x -> [id, text]
                single_result = predictor.predict(x[1])
                uncertainty = 1 - single_result["confidence"]
                
                database.save_model_prediction(
                    sample_id=x[0],
                    predicted_class=single_result["predicted_class"],
                    uncertainty_score=uncertainty
                )                

    @staticmethod
    def _calculate_kmeans_selection(samples, n_samples):
        """
        Diversity Sampling: Embedding + KMeans kullanarak seçim yapar.
        """
        if not samples:
            return []

        print("Diversity: Embedding işlemi başladı...")
        texts = [row[1] for row in samples]
        
        # Embedding çıkar
        X_pool = ActiveLearning.embedding_model.encode(texts, show_progress_bar=False)
        
        if len(X_pool) <= n_samples:
            return samples

        print(f"Diversity: KMeans ile {n_samples} adet farklı örnek seçiliyor...")
        kmeans = KMeans(n_clusters=n_samples, random_state=42, n_init="auto")
        kmeans.fit(X_pool)
        
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        
        selected_samples = []
        for cluster_id in range(n_samples):
            cluster_idxs = np.where(labels == cluster_id)[0]
            if len(cluster_idxs) == 0: continue

            pts = X_pool[cluster_idxs]
            center = centers[cluster_id]
            dists = np.linalg.norm(pts - center, axis=1)
            
            # Merkeze en yakın örneği al
            closest_idx = cluster_idxs[np.argmin(dists)]
            selected_samples.append(samples[closest_idx])
        
        return selected_samples

    @staticmethod
    def prep_labels(samples):
        """
        KRİTİK BÖLÜM: Seçilen örneklerin 'work_type' etiketini DB'den çeker.
        Kullanıcıya sormaz, DB'deki ground truth'u kullanır.
        """
        print(f"{len(samples)} adet örnek seçildi. DB'den work_type etiketleri çekiliyor...")
        
        labelled_data = [] # (text, label) çiftlerini tutacak

        for sample in samples:
            sample_id = sample[0]
            text_content = sample[1]
            
            # DB'den work_type değerini OKU (Etiketleme simülasyonu)
            # Bu metod veritabanındaki o satırın 'work_type' sütununu döndürmeli.
            true_label = database.get_ground_truth_label(sample_id, label_column="work_type")
            
            if true_label is None:
                print(f"Uyarı: ID {sample_id} için work_type boş! Atlanıyor.")
                continue

            print(f" -> ID: {sample_id} | Label bulundu: {true_label}")
            
            # Veriyi eğitim formatına uygun hale getiriyoruz
            labelled_data.append({
                "text": text_content,
                "label": true_label,
                "id": sample_id
            })
            
            # DB'de bu kaydı "artık eğitimde kullanıldı" (is_labelled=True) olarak işaretle
            database.mark_as_labelled(sample_id)

        return labelled_data

    @staticmethod
    def train_iterate(labelled_samples):
        """
        Yeni etiketlenmiş verilerle modeli eğitir.
        """
        if not labelled_samples:
            print("Eğitilecek yeni veri yok.")
            return

        print(f"Model {len(labelled_samples)} yeni örnek ile yeniden eğitiliyor...")
        
        # JobClassifierTrainer sınıfının beklediği formatı hazırla
        # Genelde (texts, labels) listeleri beklenir
        train_texts = [item["text"] for item in labelled_samples]
        train_labels = [item["label"] for item in labelled_samples]
        
        trainer = JobClassifierTrainer()
        trainer.pipeline(train_texts, train_labels)

    @staticmethod
    def diversity_sampling():
        print("--- Diversity Sampling Döngüsü Başlatılıyor ---")
        
        # 1. Havuzdan veri çek (Etiketi model tarafından BİLİNMEYEN veriler)
        # work_type dolu olsa bile, 'is_labelled' = False olanları çeker.
        pool_batch = database.get_unlabelled_samples(
            limit=ActiveLearning.hyper_params["POOL_SIZE"], 
            offset=0
        )

        if not pool_batch:
            print("Havuz boş. İşlem tamamlandı.")
            return

        # 2. KMeans ile en temsili örnekleri seç
        selected_samples = ActiveLearning._calculate_kmeans_selection(
            pool_batch, 
            ActiveLearning.hyper_params["N"]
        )
        
        # 3. DB'den bu örneklerin gerçek etiketlerini (work_type) getir
        labelled_batch = ActiveLearning.prep_labels(selected_samples)
        
        # 4. Modeli eğit
        ActiveLearning.train_iterate(labelled_batch)
        
        print("Tur tamamlandı.")

    @staticmethod
    def uncertainty_sampling(max_samples=50):
        print("--- Uncertainty Sampling Döngüsü Başlatılıyor ---")
        
        # 1. Tahmin yap ve belirsizlik skorlarını kaydet
        ActiveLearning.model_predict(max_samples)
        
        # 2. En belirsiz N örneği seç
        selected_samples = database.select_samples_to_train(ActiveLearning.hyper_params["N"])
        
        # 3. DB'den bu örneklerin gerçek etiketlerini (work_type) getir
        labelled_batch = ActiveLearning.prep_labels(selected_samples)
        
        # 4. Modeli eğit
        ActiveLearning.train_iterate(labelled_batch)
        print("Tur tamamlandı.")

if __name__ == '__main__':
    # Hangisini çalıştırmak istersen:
    ActiveLearning.diversity_sampling()
    # ActiveLearning.uncertainty_sampling()