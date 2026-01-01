# ram_database.py
import csv
import os
import json
import math
import random
import numpy as np
from collections import defaultdict

import psycopg2
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from typing import List, Tuple, Optional, Dict, Any
import pickle
import psycopg2
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

env_loaded = False

class RAMDatabase:
    """
    RAM üzerinde çalışan, CSV tabanlı veritabanı simülasyonu
    """
    
    def __init__(self, data_csv_path: str = None, load_from_csv: bool = True):
        """
        Args:
            data_csv_path: CSV dosya yolu
            load_from_csv: CSV'den yükle (True) veya boş başlat (False)
        """
        # Ana veri yapıları
        self.pool = []  # [(id, description, is_labelled, label, model_prediction, uncertainty_score), ...]
        self.test_data = []  # Test seti
        self.results = []  # Sonuç kayıtları
        self.committee_tables = {}  # Komite tabloları: {table_name: {pool_id: (prediction, uncertainty)}}
        
        # Sayaçlar
        self.next_id = 1
        self.test_id_counter = 1
        
        # CSV yolu - None değilse kullan, yoksa varsayılan
        if data_csv_path is None:
            self.data_csv_path = "./data/job_postings.csv"
        else:
            self.data_csv_path = data_csv_path
        
        # Environment yükle
        self._load_env()
        
        # Eğer CSV'den yükle
        if load_from_csv:
            balanced_csv_path = "./data/balanced_dataset.csv"
            if os.path.exists(balanced_csv_path):
                self.data_csv_path = balanced_csv_path
                self.load_from_csv(self.data_csv_path)
            else:
                raise FileNotFoundError(
                    f"Balanced dataset not found at {balanced_csv_path}. "
                    "Please run scripts/prepare_balanced_dataset.py to generate it before starting active learning."
                )
    
    @staticmethod
    def _load_env():
        global env_loaded
        if env_loaded:
            return  # tekrar yükleme!

        """Üst dizindeki .env dosyasını yükler"""
        # Mevcut dosyanın bulunduğu dizin
        current_dir = Path(__file__).parent
        
        # Bir üst dizin (ara-proje-repo)
        parent_dir = current_dir.parent
        
        # .env dosyasının tam yolu
        env_path = parent_dir / '.env'
        
        print(f".env dosyası aranıyor: {env_path}")
        print(f"Dosya var mı: {env_path.exists()}")
        
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            print(".env dosyası başarıyla yüklendi")
        else:
            print("UYARI: .env dosyası bulunamadı!")
            # Alternatif olarak kök dizinde ara
            root_env = Path.cwd() / '.env'
            if root_env.exists():
                load_dotenv(dotenv_path=root_env)
                print("Kök dizindeki .env dosyası yüklendi")
        env_loaded = True
 
    def get_db_connection(self):
        """Local DB connection"""
        return psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            port=os.getenv("DB_PORT", "5432")
        )
    
    def download_dataset(self):
        """HuggingFace'ten dataset indir"""
        try:
            print("Downloading dataset from HuggingFace...")
            from datasets import load_dataset
            dataset = load_dataset("datastax/linkedin_job_listings")
            
            # Data dizinini oluştur
            os.makedirs(os.path.dirname(self.data_csv_path), exist_ok=True)
            
            # CSV'ye kaydet
            dataset["train"].to_csv(self.data_csv_path, index=False)
            print(f"Dataset downloaded and saved to {self.data_csv_path}")

            # Optional row limiting (disabled by default). Set DATASET_ROW_LIMIT to enable.
            row_limit = os.getenv("DATASET_ROW_LIMIT")
            if row_limit and os.path.exists(self.data_csv_path):
                try:
                    row_limit_int = int(row_limit)
                    if row_limit_int > 0:
                        import pandas as pd
                        df = pd.read_csv(self.data_csv_path, nrows=row_limit_int)
                        df.to_csv(self.data_csv_path, index=False)
                        print(f"Kept first {row_limit_int} records due to DATASET_ROW_LIMIT")
                except Exception as e:
                    print(f"Error applying DATASET_ROW_LIMIT: {e}")
                    
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Creating sample data instead...")
            self.create_sample_data(1000)
    
    def load_from_csv(self, csv_path: str) -> int:
        """
        CSV dosyasından verileri yükle
        
        Returns:
            Yüklenen kayıt sayısı
        """
        loaded_count = 0
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # CSV'den gerekli alanları al
                    # Örnek: formatted_work_type'i label olarak kullan
                    description = row.get('description', '')
                    label = row.get('formatted_work_type', '')
                    
                    if description and label:
                        self.pool.append((
                            self.next_id,
                            description,
                            'FALSE',  # is_labelled
                            label,
                            None,     # model_prediction
                            None      # uncertainty_score
                        ))
                        self.next_id += 1
                        loaded_count += 1
                    
                    if loaded_count % 1000 == 0:
                        print(f"Loaded {loaded_count} records...")
            
            print(f"Successfully loaded {loaded_count} records from {csv_path}")

            # Test seti oluştur
            self.split_pool_to_test(fraction=0.2)
            
        except Exception as e:
            print(f"Error loading CSV: {e}")
            # Örnek veri oluştur
            self.create_sample_data()
        
        return loaded_count
    
    def create_sample_data(self, num_samples: int = 1000):
        """
        Örnek veri oluştur (CSV yoksa)
        """
        job_types = ['Full-time', 'Part-time', 'Contract', 'Internship', 'Temporary', 'Volunteer']
        sample_descriptions = [
            "Software engineer with Python experience",
            "Marketing manager for tech company",
            "Data scientist with machine learning skills",
            "Sales representative for enterprise software",
            "Product manager with agile experience",
            "UX designer for mobile applications",
            "DevOps engineer with cloud experience",
            "Business analyst for finance sector",
            "Customer support specialist",
            "Technical writer for API documentation"
        ]
        
        for i in range(num_samples):
            desc = random.choice(sample_descriptions)
            label = random.choice(job_types)
            self.pool.append((
                self.next_id,
                f"{desc} #{i}",
                'FALSE',
                label,
                None,
                None
            ))
            self.next_id += 1
        
        print(f"Created {num_samples} sample records")
        self.split_pool_to_test(fraction=0.2)
    
    def get_unlabelled_samples(self, batch_size: int = 1000, offset: int = 0) -> List[Tuple]:
        """
        Etiketlenmemiş örnekleri getir
        
        Args:
            batch_size: Toplu iş boyutu
            offset: Başlangıç konumu
            
        Returns:
            [(id, description, is_labelled, label, model_prediction, uncertainty_score), ...]
        """
        unlabelled = [record for record in self.pool if record[2] == 'FALSE']
        return unlabelled[offset:offset + batch_size]
    
    def get_unlabelled_with_labels(self, batch_size: Optional[int] = None, offset: int = 0) -> List[Tuple]:
        """
        Diversity sampling için: Etiketsiz örnekleri description ve label ile birlikte döner
        """
        unlabelled = [record for record in self.pool 
                     if record[2] == 'FALSE' and record[1] and record[3]]
        
        if batch_size is None:
            return unlabelled[offset:]
        else:
            return unlabelled[offset:offset + batch_size]
    
    def get_all_unlabelled_with_labels(self) -> List[Tuple]:
        """
        Tüm etiketsiz örnekleri description ve label ile birlikte döner
        """
        return [record for record in self.pool 
                if record[2] == 'FALSE' and record[1] and record[3]]
    
    def save_model_prediction(self, sample_id: int, predicted_class: str, uncertainty_score: float):
        """
        Model tahminini kaydet
        """
        for i, record in enumerate(self.pool):
            if record[0] == sample_id:
                # Tuple immutable olduğu için yeni tuple oluştur
                self.pool[i] = (
                    record[0],
                    record[1],
                    record[2],  # is_labelled
                    record[3],  # label
                    predicted_class,
                    str(uncertainty_score) if uncertainty_score is not None else None
                )
                break
    
    def update_labelled_sample(self, sample_id: int, label: str):
        """
        Örneği etiketli olarak işaretle
        """
        for i, record in enumerate(self.pool):
            if record[0] == sample_id:
                self.pool[i] = (
                    record[0],
                    record[1],
                    'TRUE',  # is_labelled
                    label,
                    record[4],  # model_prediction
                    record[5]   # uncertainty_score
                )
                break
    
    def uncertainty_sampling_selection(self, N: int = 100) -> List[Tuple]:
        """
        Belirsizlik skoruna göre örnek seçimi
        """
        # Tahmin yapılmış ve belirsizlik skoru olan örnekleri filtrele
        candidates = []
        for record in self.pool:
            if (record[2] == 'FALSE' and  # is_labelled = FALSE
                record[4] is not None and  # model_prediction var
                record[5] is not None):    # uncertainty_score var
                
                try:
                    uncertainty = float(record[5])
                    candidates.append((record, uncertainty))
                except (ValueError, TypeError):
                    continue
        
        # Belirsizliğe göre sırala (yüksek belirsizlik önce)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # İlk N örneği seç
        selected = [candidate[0] for candidate in candidates[:N]]
        return selected
    
    def diversity_sampling_selection(self, N: int = 100, 
                                     embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                                     max_samples: Optional[int] = None) -> List[Tuple]:
        """
        Diversity sampling ile örnek seçer (KMeans tabanlı)
        """
        # Etiketsiz örnekleri al
        if max_samples is not None:
            samples = self.get_unlabelled_with_labels(batch_size=max_samples, offset=0)
        else:
            samples = self.get_all_unlabelled_with_labels()
        
        if not samples or len(samples) < N:
            return samples[:N] if samples else []
        
        ids = [s[0] for s in samples]
        descriptions = [s[1] for s in samples]
        labels = [s[3] for s in samples]
        
        # Embedding hesapla
        print(f"Diversity sampling: {len(descriptions)} örnek için embedding hesaplanıyor...")
        try:
            model = SentenceTransformer(embedding_model_name)
            embeddings = model.encode(descriptions, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
        except Exception as e:
            print(f"Embedding error, using random sampling: {e}")
            # Fallback: random sampling
            random_indices = random.sample(range(len(samples)), min(N, len(samples)))
            selected_samples = []
            for idx in random_indices:
                selected_samples.append((
                    ids[idx],
                    descriptions[idx],
                    'FALSE',
                    labels[idx],
                    None,
                    None
                ))
            return selected_samples
        
        # KMeans ile diversity sampling
        n_samples = min(N, len(embeddings))
        if len(embeddings) <= n_samples:
            selected_indices = list(range(len(embeddings)))
        else:
            try:
                kmeans = KMeans(n_clusters=n_samples, random_state=42, n_init="auto")
                kmeans.fit(embeddings)
                cluster_labels = kmeans.labels_
                centers = kmeans.cluster_centers_
                
                selected_indices = []
                for cluster_id in range(n_samples):
                    idxs = np.where(cluster_labels == cluster_id)[0]
                    if len(idxs) == 0:
                        continue
                    
                    pts = embeddings[idxs]
                    center = centers[cluster_id]
                    dist = np.linalg.norm(pts - center, axis=1)
                    closest_idx = idxs[np.argmin(dist)]
                    selected_indices.append(closest_idx)
                
                selected_indices = np.array(selected_indices)
            except Exception as e:
                print(f"KMeans error: {e}, using random selection")
                selected_indices = np.random.choice(len(embeddings), size=n_samples, replace=False)
        
        # Seçilen örnekleri döndür
        selected_samples = []
        for idx in selected_indices:
            selected_samples.append((
                ids[idx],
                descriptions[idx],
                'FALSE',
                labels[idx],
                None,
                None
            ))
        
        print(f"Diversity sampling: {len(selected_samples)} örnek seçildi.")
        return selected_samples
    
    def get_all_uncertainty_scores(self) -> List[float]:
        """
        Tüm uncertainty skorlarını getir
        """
        scores = []
        for record in self.pool:
            if record[5] is not None:  # uncertainty_score
                try:
                    scores.append(float(record[5]))
                except (ValueError, TypeError):
                    continue
        return scores
    
    def get_labeled_samples(self, limit: Optional[int] = None) -> List[Tuple]:
        """
        Etiketli örnekleri getir
        """
        labelled = [record for record in self.pool if record[2] == 'TRUE' and record[1] and record[3]]
        
        if limit is not None:
            return labelled[:limit]
        return labelled
    
    def get_test_samples(self, limit: Optional[int] = None, offset: int = 0) -> List[Tuple]:
        """
        Test setinden örnekleri getir
        """
        if limit is None:
            return self.test_data[offset:]
        else:
            return self.test_data[offset:offset + limit]
    
    def split_pool_to_test(self, fraction: float = 0.2, seed: int = 42):
        """
        Pool'dan rastgele test seti oluştur
        """
        random.seed(seed)
        
        # Etiketli ve açıklaması olan örnekleri al
        candidates = [record for record in self.pool if record[1] and record[3]]
        
        if not candidates:
            return {"moved_count": 0, "moved_ids": []}
        
        # Kaç örnek taşınacak
        k = int(len(candidates) * fraction)
        if k <= 0:
            return {"moved_count": 0, "moved_ids": []}
        
        # Rastgele örnek seç
        selected_records = random.sample(candidates, k)
        selected_ids = [record[0] for record in selected_records]
        
        # Test setine ekle
        self.test_data.extend(selected_records)
        
        # Pool'dan çıkar
        self.pool = [record for record in self.pool if record[0] not in selected_ids]
        
        return {"moved_count": k, "moved_ids": selected_ids}
    
    def reset_pool(self, clear_labels: bool = True, clear_predictions: bool = True):
        """
        Pool'u sıfırla
        """
        counts = {"labels_cleared": 0, "predictions_cleared": 0}
        
        for i, record in enumerate(self.pool):
            new_record = list(record)
            
            if clear_labels:
                new_record[2] = 'FALSE'  # is_labelled
                new_record[3] = None     # label
                counts["labels_cleared"] += 1
            
            if clear_predictions:
                new_record[4] = None     # model_prediction
                new_record[5] = None     # uncertainty_score
                counts["predictions_cleared"] += 1
            
            self.pool[i] = tuple(new_record)
        
        return counts
    
    def initialize_labeled_pool(self, initial_size: int = 100):
        """
        Başlangıç için rastgele örnekleri etiketli olarak işaretle
        """
        # Etiketlenmemiş ve etiketi olan örnekleri al
        unlabelled = [record for record in self.pool 
                     if record[2] == 'FALSE' and record[3] is not None]
        
        if not unlabelled:
            return
        
        # Kaç örnek etiketlenecek
        # Küçük datasetlerde tamamını labeled yapmak active learning'i kilitler.
        # Bu yüzden initial labeled oranını maksimum %10 ile sınırla.
        max_fraction = 0.10
        fraction_cap = max(1, int(len(unlabelled) * max_fraction))
        n = min(initial_size, len(unlabelled), fraction_cap)
        
        # Rastgele seç
        selected = random.sample(unlabelled, n)
        selected_ids = [record[0] for record in selected]
        
        # Etiketli olarak işaretle
        for sample_id in selected_ids:
            self.update_labelled_sample(sample_id, self.get_label_by_id(sample_id))
        
        print(f"Başlangıç için {n} örnek labeled olarak işaretlendi.")
    
    def get_label_by_id(self, sample_id: int) -> Optional[str]:
        """
        ID'ye göre etiketi getir
        """
        for record in self.pool:
            if record[0] == sample_id:
                return record[3]
        return None
    
    def get_next_test_id(self) -> int:
        """
        Bir sonraki test ID'sini getir
        """
        test_id = self.test_id_counter
        self.test_id_counter += 1
        return test_id
    
    def insert_test_result(self, 
                          test_id: str,
                          iteration_no: int,
                          model_name: str,
                          train_data_size: int,
                          method: str = None,
                          data_size: int = None,
                          N: int = None,
                          T: int = None,
                          I: int = None,
                          metrics: Dict = None,
                          params: Dict = None,
                          run_by: str = None,
                          notes: str = None) -> int:
        """
        Test sonucunu local RAM'e kaydet
        """
        result_id = len(self.results) + 1
        self.results.append({
            'id': result_id,
            'test_id': test_id,
            'iteration_no': iteration_no,
            'model_name': model_name,
            'train_data_size': train_data_size,
            'method': method,
            'data_size': data_size,
            'n': N,
            't': T,
            'i': I,
            'metrics': metrics,
            'params': params,
            'run_by': run_by,
            'notes': notes
        })
        
        return result_id
    
    def insert_test_result_remote(self, test_id: str,
                                  iteration_no: int,
                                  model_name: str,
                                  train_data_size: int,
                                  method: str = None,
                                  data_size: int = None,
                                  N: int = None,
                                  T: int = None,
                                  I: int = None,
                                  metrics: dict = None,
                                  params: dict = None,
                                  run_by: str = None,
                                  notes: str = None):
        """Insert test result into an online DB using DSN in `NEON_API_KEY` or `DATABASE_URL`.

        Returns the inserted ID if successful, None otherwise.
        """
        try:
            dsn = os.getenv("NEON_API_KEY") or os.getenv("DATABASE_URL")
            if not dsn:
                print("No NEON_API_KEY or DATABASE_URL found in environment. Skipping remote insert.")
                return None

            # ensure local results table exists schema-wise on remote as well
            create_sql = """
            CREATE TABLE IF NOT EXISTS results (
                id SERIAL PRIMARY KEY,
                test_id TEXT,
                iteration_no INTEGER,
                model_name TEXT,
                train_data_size INTEGER,
                method TEXT,
                data_size INTEGER,
                n INTEGER,
                t INTEGER,
                i INTEGER,
                metrics JSONB,
                params JSONB,
                run_by TEXT,
                notes TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
            );
            """

            insert_sql = """
            INSERT INTO results (test_id, iteration_no, model_name, train_data_size, method, data_size, n, t, i, metrics, params, run_by, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id;
            """

            metrics_json = json.dumps(metrics) if metrics is not None else None
            params_json = json.dumps(params) if params is not None else None

            with psycopg2.connect(dsn) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(create_sql)
                    cursor.execute(insert_sql, (
                        test_id,
                        iteration_no,
                        model_name,
                        train_data_size,
                        method,
                        data_size,
                        N,
                        T,
                        I,
                        metrics_json,
                        params_json,
                        run_by,
                        notes
                    ))
                    inserted = cursor.fetchone()
                conn.commit()

            inserted_id = inserted[0] if inserted else None
            print(f"Successfully inserted to remote DB with ID: {inserted_id}")
            return inserted_id
            
        except Exception as e:
            print(f"Error inserting to remote DB: {e}")
            return None
    
    def save_to_csv(self, pool_path: str = "./data/pool.csv", 
                    test_path: str = "./data/test_data.csv",
                    results_path: str = "./data/results.csv"):
        """
        Verileri CSV'ye kaydet
        """
        os.makedirs(os.path.dirname(pool_path), exist_ok=True)
        
        # Pool'u kaydet
        with open(pool_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'description', 'is_labelled', 'label', 
                           'model_prediction', 'uncertainty_score'])
            writer.writerows(self.pool)
        
        # Test verisini kaydet
        with open(test_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'description', 'is_labelled', 'label', 
                           'model_prediction', 'uncertainty_score'])
            writer.writerows(self.test_data)
        
        # Sonuçları kaydet
        if self.results:
            with open(results_path, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['id', 'test_id', 'iteration_no', 'model_name', 
                            'train_data_size', 'method', 'data_size', 'n', 't', 'i',
                            'metrics', 'params', 'run_by', 'notes']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.results)
        
        print(f"Data saved to CSV files")

# Global database instance - Varsayılan path ile oluştur
database = RAMDatabase(data_csv_path="./data/job_postings.csv", load_from_csv=True)

if __name__ == '__main__':
    # Test
    db = RAMDatabase()
    print(f"Loaded {len(db.pool)} records to pool")
    print(f"Test set: {len(db.test_data)} records")
    print(f"Labelled samples: {len(db.get_labeled_samples())}")