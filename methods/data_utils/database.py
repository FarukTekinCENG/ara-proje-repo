import psycopg2
import os
import sys

class Database:
    def __init__(self):
        # ---------------------------------------------------------
        # DB BAĞLANTI AYARLARI
        # Burayı kendi DBeaver/Postgres bilgilerine göre düzenle
        # ---------------------------------------------------------
        self.db_params = {
            "dbname": "Active_Learning",  # Senin DB adın
            "user": "postgres",           # Kullanıcı adın
            "password": "123",            # Şifren
            "host": "localhost",
            "port": "5432"
        }
        self.connection = None
        self.connect()

    def connect(self):
        try:
            self.connection = psycopg2.connect(**self.db_params)
            # print("DB Bağlantısı başarılı.")
        except Exception as e:
            print(f"DB Bağlantı Hatası: {e}")
            sys.exit(1)

    def get_cursor(self):
        if self.connection.closed:
            self.connect()
        return self.connection.cursor()

    # ---------------------------------------------------------
    # 1. VERİ ÇEKME METODLARI
    # ---------------------------------------------------------
    def get_unlabelled_samples(self, limit=1000, offset=0):
        """
        Henüz eğitimde kullanılmamış (is_labelled = FALSE) verileri çeker.
        Dönüş formatı: [(id, description), (id, description), ...]
        """
        query = """
            SELECT id, description 
            FROM postings 
            WHERE is_labelled = FALSE 
            AND description IS NOT NULL
            ORDER BY id  -- Sayfalama (pagination) tutarlılığı için sıralama şart
            LIMIT %s OFFSET %s
        """
        try:
            with self.get_cursor() as cursor:
                cursor.execute(query, (limit, offset))
                results = cursor.fetchall()
            return results
        except Exception as e:
            print(f"Hata (get_unlabelled_samples): {e}")
            return []

    def get_ground_truth_label(self, sample_id, label_column="work_type"):
        """
        Simülasyon için: Verilen ID'nin gerçek etiketini DB'den okur.
        """
        query = f"SELECT {label_column} FROM postings WHERE id = %s"
        try:
            with self.get_cursor() as cursor:
                cursor.execute(query, (sample_id,))
                result = cursor.fetchone()
                if result:
                    return result[0]
                return None
        except Exception as e:
            print(f"Hata (get_ground_truth_label): {e}")
            return None

    # ---------------------------------------------------------
    # 2. UNCERTAINTY SAMPLING İÇİN METODLAR
    # ---------------------------------------------------------
    def save_model_prediction(self, sample_id, predicted_class, uncertainty_score):
        """
        Modelin tahminini ve belirsizlik skorunu DB'ye yazar.
        Tablonda 'uncertainty_score' diye bir sütun açtığını varsayıyoruz.
        """
        query = """
            UPDATE postings 
            SET uncertainty_score = %s 
            WHERE id = %s
        """
        try:
            with self.get_cursor() as cursor:
                cursor.execute(query, (uncertainty_score, sample_id))
            self.connection.commit()
        except Exception as e:
            print(f"Hata (save_model_prediction): {e}")
            self.connection.rollback()

    def select_samples_to_train(self, n=5):
        """
        Uncertainty Sampling: En yüksek belirsizlik skoruna sahip
        ve henüz etiketlenmemiş (is_labelled=False) N kaydı getirir.
        """
        query = """
            SELECT id, description 
            FROM postings 
            WHERE is_labelled = FALSE 
            AND uncertainty_score IS NOT NULL
            ORDER BY uncertainty_score DESC 
            LIMIT %s
        """
        try:
            with self.get_cursor() as cursor:
                cursor.execute(query, (n,))
                results = cursor.fetchall()
            return results
        except Exception as e:
            print(f"Hata (select_samples_to_train): {e}")
            return []

    # ---------------------------------------------------------
    # 3. GÜNCELLEME METODLARI
    # ---------------------------------------------------------
    def mark_as_labelled(self, sample_id):
        """
        Veriyi eğitim setine dahil edildi olarak işaretler.
        Böylece bir sonraki turda tekrar seçilmez.
        """
        query = "UPDATE postings SET is_labelled = TRUE WHERE id = %s"
        try:
            with self.get_cursor() as cursor:
                cursor.execute(query, (sample_id,))
            self.connection.commit()
        except Exception as e:
            print(f"Hata (mark_as_labelled): {e}")
            self.connection.rollback()

    def update_sample_label(self, sample_id, label_column, label_value):
        """
        Eğer etiketi manuel güncelliyorsak kullanılır.
        """
        query = f"UPDATE postings SET {label_column} = %s, is_labelled = TRUE WHERE id = %s"
        try:
            with self.get_cursor() as cursor:
                cursor.execute(query, (label_value, sample_id))
            self.connection.commit()
        except Exception as e:
            print(f"Hata (update_sample_label): {e}")
            self.connection.rollback()

# Singleton Instance
# Diğer dosyalardan 'from data_utils.database import database' diyerek buna erişirsin.
database = Database()