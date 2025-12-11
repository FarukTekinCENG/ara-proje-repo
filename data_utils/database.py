import psycopg2
import os
import sys
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
              AND label IS NOT NULL
              AND model_prediction IS DISTINCT FROM label
            ORDER BY uncertainty_score DESC
            LIMIT %s;
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
    def select_samples_to_train(N=100):
        # (SELECT is_labelled=FALSE) AND
        # (model_prediction != label) AND
        # (ORDER BY uncertainty_score DESC) AND
        # (LIMIT = N)        
        query = database.query_list[3]
        with database.get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (N,))
                return cursor.fetchall()

if __name__ == '__main__':
    database.run_query(2)
    print(database.get_unlabelled_samples())
