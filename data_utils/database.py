import psycopg2
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

env_loaded = False

class database:
    query_list = {
        1: "SELECT * FROM pool WHERE is_labelled = 'FALSE' ORDER BY id LIMIT %s OFFSET %s;",  # samples with no label
        2: "UPDATE pool SET model_prediction = %s, uncertainty_score = %s WHERE id = %s;",
        3: """
        SELECT *
        FROM pool
        WHERE is_labelled = 'FALSE'
          AND model_prediction IS NOT NULL
        ORDER BY uncertainty_score DESC
        LIMIT %s;
        """,
        4: """
        SELECT COUNT(*)
        FROM pool
        WHERE is_labelled = 'FALSE';
        
        """
    }

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
 
    @staticmethod
    def get_db_connection():
        database._load_env()
        return psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            port=os.getenv("DB_PORT", "5432")
        )
    
    @staticmethod
    def ensure_results_table():
        """Create a `results` table to store model test runs.

        Columns:
        - id: serial primary key
        - test_id: identifier for the test run (string)
        - iteration_no: iteration number
        - model_name: model identifier
        - train_data_size: number of training samples
        - n, t, i: hyperparameters N, T, I (nullable integers)
        - metrics: JSONB field to store evaluation metrics (accuracy, f1, etc.)
        - params: JSONB field to store other params (DATA_SIZE, etc.)
        - run_by: user or script that ran the test
        - notes: free text notes
        - created_at: timestamp
        """
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

        with database.get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(create_sql)
            conn.commit()

    @staticmethod
    def insert_test_result(test_id: str,
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
        """Insert a test result row into `results` table and return inserted id.

        Example:
            database.insert_test_result(
                test_id='exp-1', iteration_no=0, model_name='eurobert',
                train_data_size=1000, N=100, T=10, I=5,
                metrics={'accuracy':0.92}, params={'DATA_SIZE':1000}, run_by='script'
            )
        """
        # ensure table exists
        database.ensure_results_table()

        insert_sql = """
        INSERT INTO results (test_id, iteration_no, model_name, train_data_size, method, data_size, n, t, i, metrics, params, run_by, notes)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
        """

        metrics_json = json.dumps(metrics) if metrics is not None else None
        params_json = json.dumps(params) if params is not None else None

        with database.get_db_connection() as conn:
            with conn.cursor() as cursor:
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

        return inserted[0] if inserted else None

    @staticmethod
    def insert_test_result_remote(test_id: str,
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

        This does not affect the local DB connection behavior.
        """
        dsn = os.getenv("NEON_API_KEY") or os.getenv("DATABASE_URL")
        if not dsn:
            raise RuntimeError("No NEON_API_KEY or DATABASE_URL found in environment for remote insert")

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

        return inserted[0] if inserted else None

    @staticmethod
    def get_next_test_id():
        """Return next numeric test_id as int (max existing numeric test_id + 1).

        Looks only at numeric `test_id` values in `results`. Returns 1 if none.
        """
        database.ensure_results_table()
        query = """
        SELECT MAX(CAST(test_id AS INTEGER))
        FROM results
        WHERE test_id ~ '^[0-9]+$';
        """
        with database.get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                row = cursor.fetchone()
        max_id = row[0] if row and row[0] is not None else 0
        return int(max_id) + 1
    
    @staticmethod
    def run_query(selection=1):
        query=database.query_list[selection]
        with database.get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                result = cursor.fetchone()
                # print("query result: ", result)
                return result

    @staticmethod
    def get_unlabelled_samples(batch_size, offset):
        with database.get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(database.query_list[1], (batch_size, offset))
                results = cursor.fetchall()
                return results

    @staticmethod
    def save_model_prediction(sample_id: int, predicted_class: str, uncertainty_score: float):
        query = database.query_list[2]
        with database.get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (predicted_class, uncertainty_score, sample_id))
            conn.commit()

    @staticmethod
    def uncertainty_sampling_selection(N=100):      
        query = database.query_list[3]
        with database.get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (N,))
                return cursor.fetchall()

    @staticmethod
    def diversity_sampling_selection(N=100, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2", max_samples=None):
        """
        Diversity sampling ile örnek seçer
        KMeans tabanlı kümeleme kullanır
        
        Args:
            N: Seçilecek örnek sayısı
            embedding_model_name: Embedding model adı
            max_samples: Embedding oluşturulacak maksimum örnek sayısı (None ise tüm veri kullanılır)
        """
        import numpy as np
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import KMeans
        
        # Etiketsiz örnekleri çek - max_samples parametresi ile sınırlandır
        if max_samples is not None:
            # Sadece belirtilen kadar örnek çek
            samples = database.get_unlabelled_with_labels(batch_size=max_samples, offset=0)
            print(f"Diversity sampling: max_samples={max_samples} parametresi ile sınırlandırıldı.")
        else:
            # Tüm örnekleri çek (eski davranış)
            samples = database.get_all_unlabelled_with_labels()
            print(f"Diversity sampling: Tüm etiketsiz örnekler kullanılıyor (max_samples belirtilmedi).")
        
        if not samples or len(samples) < N:
            # Eğer yeterli örnek yoksa, mevcut olanları döndür
            return samples[:N] if samples else []
        
        ids = [s[0] for s in samples]
        descriptions = [s[1] for s in samples]
        labels = [s[2] for s in samples]
        
        # Embedding hesapla
        print(f"Diversity sampling: {len(descriptions)} örnek için embedding hesaplanıyor...")
        model = SentenceTransformer(embedding_model_name)
        embeddings = model.encode(descriptions, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
        
        # KMeans ile diversity sampling
        n_samples = min(N, len(embeddings))
        if len(embeddings) <= n_samples:
            selected_indices = np.arange(len(embeddings))
        else:
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
        
        # Seçilen örnekleri döndür (uncertainty_sampling_selection ile aynı format)
        # Format: (id, description, is_labelled, label, model_prediction, uncertainty_score)
        selected_samples = []
        for idx in selected_indices:
            selected_samples.append((
                ids[idx],
                descriptions[idx],
                'FALSE',  # is_labelled
                labels[idx],  # label
                None,  # model_prediction
                None   # uncertainty_score
            ))
        
        print(f"Diversity sampling: {len(selected_samples)} örnek seçildi.")
        return selected_samples
    
    @staticmethod
    def is_all_labelled():
        query = database.query_list[4]
        with database.get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                return cursor.fetchall()

    @staticmethod
    def get_all_uncertainty_scores():
        """
        Pool'daki tüm model tahminlerinin uncertainty_score'larını liste olarak döner
        """
        query = """
            SELECT uncertainty_score
            FROM pool
            WHERE model_prediction IS NOT NULL;
        """
        with database.get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                results = cursor.fetchall()
                # results [(0.23,), (0.1,), ...] şeklinde gelir, sadece değerleri al
                scores = [row[0] for row in results]
                return scores

    @staticmethod
    def update_labelled_sample(sample_id, label):
        query = """
            UPDATE pool
            SET label = %s, is_labelled = 'TRUE'
            WHERE id = %s;
        """
        with database.get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (label, sample_id))
            conn.commit()

    @staticmethod
    def get_unlabelled_with_labels(batch_size=None, offset=0):
        """
        Diversity sampling için: Etiketsiz örnekleri description ve label ile birlikte döner
        (label zaten var, sadece is_labelled = 'FALSE')
        """
        if batch_size is None:
            query = """
                SELECT id, description, label
                FROM pool
                WHERE is_labelled = 'FALSE'
                  AND description IS NOT NULL
                  AND label IS NOT NULL
                ORDER BY id
                OFFSET %s;
            """
            params = (offset,)
        else:
            query = """
                SELECT id, description, label
                FROM pool
                WHERE is_labelled = 'FALSE'
                  AND description IS NOT NULL
                  AND label IS NOT NULL
                ORDER BY id
                LIMIT %s OFFSET %s;
            """
            params = (batch_size, offset)
        
        with database.get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                return cursor.fetchall()

    @staticmethod
    def get_all_unlabelled_with_labels():
        """
        Tüm etiketsiz örnekleri description ve label ile birlikte döner
        """
        query = """
            SELECT id, description, label
            FROM pool
            WHERE is_labelled = 'FALSE'
              AND description IS NOT NULL
              AND label IS NOT NULL
            ORDER BY id;
        """
        with database.get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                return cursor.fetchall()

    @staticmethod
    def get_labelled_samples():
        """
        Etiketli örnekleri döner (eğitim için)
        """
        query = """
            SELECT id, description, label
            FROM pool
            WHERE is_labelled = 'TRUE'
              AND description IS NOT NULL
              AND label IS NOT NULL
            ORDER BY id;
        """
        with database.get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                return cursor.fetchall()

    @staticmethod
    def mark_samples_as_labelled(sample_ids):
        """
        Seçilen örnekleri etiketli olarak işaretle (label zaten var)
        """
        if not sample_ids:
            return
        
        placeholders = ','.join(['%s'] * len(sample_ids))
        query = f"""
            UPDATE pool
            SET is_labelled = 'TRUE'
            WHERE id IN ({placeholders});
        """
        with database.get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, tuple(sample_ids))
            conn.commit()

    @staticmethod
    def initialize_labeled_pool(initial_size=100, random_seed=42):
        """
        Başlangıç için rastgele bir kısmı labeled olarak işaretle
        """
        # PostgreSQL'de RANDOM() kullan (küçük setler için sorun yok)
        query = """
            UPDATE pool
            SET is_labelled = 'TRUE'
            WHERE id IN (
                SELECT id 
                FROM pool 
                WHERE is_labelled = 'FALSE'
                  AND description IS NOT NULL
                  AND label IS NOT NULL
                ORDER BY RANDOM()
                LIMIT %s
            );
        """
        with database.get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (initial_size,))
                affected_rows = cursor.rowcount
            conn.commit()
        print(f"Başlangıç için {affected_rows} örnek labeled olarak işaretlendi.")

    @staticmethod
    def get_unlabelled_count():
        """
        Etiketsiz örnek sayısını döner
        """
        query = """
            SELECT COUNT(*)
            FROM pool
            WHERE is_labelled = 'FALSE'
              AND description IS NOT NULL
              AND label IS NOT NULL;
        """
        with database.get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                result = cursor.fetchone()
                return result[0] if result else 0

    @staticmethod
    def ensure_test_table():
        """Create `test_data` table to store the fixed test set copied from `pool`."""
        create_sql = """
        CREATE TABLE IF NOT EXISTS test_data (
            pool_id INTEGER PRIMARY KEY,
            description TEXT,
            is_labelled TEXT,
            label TEXT,
            model_prediction TEXT,
            uncertainty_score TEXT,
            moved_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        );
        """
        with database.get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(create_sql)
            conn.commit()

    @staticmethod
    def split_pool_to_test(fraction=0.2, seed=42):
        """Randomly split `pool` and move `fraction` portion into `test_data`.

        Returns dict with keys `moved_count` and `moved_ids`.
        """
        import random

        database.ensure_test_table()

        with database.get_db_connection() as conn:
            with conn.cursor() as cursor:
                # get eligible ids
                cursor.execute("SELECT id FROM pool WHERE description IS NOT NULL")
                rows = cursor.fetchall()
                ids = [r[0] for r in rows]
                total = len(ids)
                k = int(total * fraction)
                if k <= 0:
                    return {"moved_count": 0, "moved_ids": []}

                random.seed(seed)
                selected = random.sample(ids, k)

                placeholders = ','.join(['%s'] * len(selected))

                insert_sql = f"""
                INSERT INTO test_data (pool_id, description, is_labelled, label, model_prediction, uncertainty_score)
                SELECT id, description, is_labelled, label, model_prediction, uncertainty_score
                FROM pool
                WHERE id IN ({placeholders});
                """
                cursor.execute(insert_sql, tuple(selected))

                delete_sql = f"DELETE FROM pool WHERE id IN ({placeholders});"
                cursor.execute(delete_sql, tuple(selected))

            conn.commit()

        return {"moved_count": len(selected), "moved_ids": selected}

    @staticmethod
    def get_test_samples(limit=None, offset=0):
        """Return test samples as tuples in the same format as pool selections:
        (pool_id, description, is_labelled, label, model_prediction, uncertainty_score)
        """
        database.ensure_test_table()
        if limit is None:
            query = "SELECT pool_id, description, is_labelled, label, model_prediction, uncertainty_score FROM test_data ORDER BY pool_id OFFSET %s;"
            params = (offset,)
        else:
            query = "SELECT pool_id, description, is_labelled, label, model_prediction, uncertainty_score FROM test_data ORDER BY pool_id LIMIT %s OFFSET %s;"
            params = (limit, offset)

        with database.get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                return cursor.fetchall()

    @staticmethod
    def reset_pool(clear_labels=True, clear_predictions=True):
        """Reset pool state for a fresh run.

        - If `clear_labels` True: set `label` = NULL and `is_labelled` = 'FALSE'
        - If `clear_predictions` True: set `model_prediction` = NULL and `uncertainty_score` = NULL

        Returns dict with counts of affected rows.
        """
        counts = {"labels_cleared": 0, "predictions_cleared": 0}
        with database.get_db_connection() as conn:
            with conn.cursor() as cursor:
                if clear_labels:
                    cursor.execute("UPDATE pool SET label = NULL, is_labelled = 'FALSE' WHERE description IS NOT NULL;")
                    counts["labels_cleared"] = cursor.rowcount
                if clear_predictions:
                    cursor.execute("UPDATE pool SET model_prediction = NULL, uncertainty_score = NULL WHERE model_prediction IS NOT NULL OR uncertainty_score IS NOT NULL;")
                    counts["predictions_cleared"] = cursor.rowcount
            conn.commit()
        return counts


if __name__ == '__main__':
    database.run_query(2)
    print(database.get_unlabelled_samples())
    print(database.is_all_labelled())
