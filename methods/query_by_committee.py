import os
import sys
from collections import Counter

# ------------------------------
# Proje kök dizini path ayarı
# ------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from train import JobClassifierTrainer
from model import ModelPredictor
from data_utils.database import database


class CommitteePredictor:
    """
    Birden fazla ModelPredictor'dan oluşan komite.
    Query by Committee için her sample'a birden fazla tahmin üretir.
    """

    def __init__(self, model_paths):
        """
        model_paths: [ "./fine_tuned_eurobert_seed1", "./fine_tuned_eurobert_seed2", ... ]
        """
        self.members = [ModelPredictor(path) for path in model_paths]

    def predict_labels(self, text):
        """
        Her bir komite üyesinden predicted_class döndürür.
        Örn: ["FULL_TIME", "REMOTE", "FULL_TIME"]
        """
        labels = []
        for model in self.members:
            out = model.predict(text)
            labels.append(out["predicted_class"])
        return labels

    @staticmethod
    def disagreement_score(labels):
        """
        Basit vote-based disagreement:
        disagreement = 1 - (en çok oyu alan sınıfın oranı)

        labels = ["FULL_TIME", "REMOTE", "FULL_TIME"]
        majority = FULL_TIME (2/3)
        disagreement = 1 - 2/3 = 1/3
        """
        if not labels:
            return 0.0

        counts = Counter(labels)
        majority_count = max(counts.values())
        total = len(labels)
        disagreement = 1.0 - (majority_count / total)
        return disagreement

    @staticmethod
    def majority_label(labels):
        """
        En çok oy alan sınıfı döndür.
        """
        if not labels:
            return None
        counts = Counter(labels)
        return counts.most_common(1)[0][0]


class CommitteeActiveLearning:
    """
    Query by Committee (QBC) tabanlı aktif öğrenme akışı.
    Mimarisi, senin Uncertainty ActiveLearning sınıfına paralel:

        - committee_predict: havuzdaki örnekler için komite tahmini
        - database.save_committee_prediction ile DB'ye yaz
        - database.select_samples_to_train_by_qbc ile en yüksek disagreement'li N örnek çek
        - prep_labels -> train_iterate -> check_stop_condition
    """

    hyper_params = {
        "N": 5,       # her iterasyonda seçilecek örnek sayısı
        "I": 0.001,   # min. iyileşme (stop condition için)
        "T": 0.85     # model güven eşiği (stop condition için)
    }

    # Komite modellerinin path'lerini burada tanımlıyorsun
    committee_model_paths = [
        "./fine_tuned_eurobert_seed1",
        "./fine_tuned_eurobert_seed2",
        "./fine_tuned_eurobert_seed3"
    ]

    @staticmethod
    def committee_predict(max_samples=None):
        """
        Havuzdaki is_labelled = FALSE örnekler için komite tahminleri üretir.
        Her örnek için:

            - committee labels: ["FULL_TIME", "REMOTE", "FULL_TIME"]
            - majority_label: "FULL_TIME"
            - disagreement_score: 1 - 2/3 = 0.333...

        Bunları DB'ye yazar. Bu fonksiyon sadece
        'model tahmini + skor DB'ye yaz' işini yapar,
        seçim işini QBC sampling fonksiyonuna bırakır.

        Beklenen database fonksiyonları (örnek API):

            database.get_unlabelled_samples(batch_size, offset)
              -> [(sample_id, text, ...), ...]

            database.save_committee_prediction(
                sample_id: int,
                predicted_class: str,
                disagreement_score: float
            )
        """

        committee = CommitteePredictor(CommitteeActiveLearning.committee_model_paths)

        batch_size = 1000
        offset = 0
        total_processed = 0
        page = 0

        while True:
            if max_samples is not None and total_processed >= max_samples:
                print(f"Max samples limit reached: {max_samples}")
                break

            offset = page * batch_size
            batch = database.get_unlabelled_samples(batch_size, offset)

            if not batch:
                print("No more unlabelled samples in pool.")
                break

            # Gerekirse son batch'i trim et
            if max_samples is not None:
                remaining = max_samples - total_processed
                if len(batch) > remaining:
                    batch = batch[:remaining]

            print(f"[CommitteePredict] Page {page}: {len(batch)} records")
            total_processed += len(batch)
            page += 1

            for row in batch:
                sample_id = row[0]
                text = row[1]

                labels = committee.predict_labels(text)
                disagreement = committee.disagreement_score(labels)
                majority = CommitteePredictor.majority_label(labels)

                print(f"Sample {sample_id} - majority: {majority}, disagreement: {disagreement:.4f}")

                # DB'ye yaz: bu fonksiyonu sen data_utils.database içinde implement edeceksin
                database.save_committee_prediction(
                    sample_id=sample_id,
                    predicted_class=majority,
                    disagreement_score=disagreement
                )

    @staticmethod
    def prep_labels(samples):
        """
        Gerçek senaryoda bu aşamada:
          - İnsan anotasyonu
          - Veya LLM ile label önerisi + insan onayı
          - Sonra DB'de is_labelled = TRUE, label = ... update edilir.

        Şimdilik sadece placeholder:
        """
        print("now label your suggested samples...")
        # Örnek: burada samples içinden ID'leri alıp DB'de label'lama yapabilirsin
        return samples

    @staticmethod
    def train_iterate(samples, labels=None):
        """
        Her iterasyonda modeli yeniden eğitmek için ortak pipeline.
        samples: genelde (id, text, label, ...) tuple listesi olabilir.
        labels: istersen ayrıca pas edebilirsin.
        """
        trainer = JobClassifierTrainer()
        trainer.pipeline(samples, labels)

    @staticmethod
    def check_stop_condition():
        """
        Burayı sonradan dolduracaksın.

        Örnek kriterler:

          - Havuzda unlabelled sample kalmaması
          - Son 2 turda validation accuracy artışı I'den küçük
          - Model confidence T threshold'unu geçmiş

        Şimdilik döngü devam etsin diye 'unmet' döndürüyoruz.
        """
        return 'unmet'

    @staticmethod
    def query_by_committee_sampling(max_samples=None):
        """
        Query by Committee stratejisi ile aktif öğrenme döngüsü.

        Akış:

            while not stop:
                1) Komite tahmini yap (committee_predict)
                   -> DB'de her unlabelled sample için disagreement_score güncellensin
                2) DB'den en yüksek disagreement'li N sample çek
                   (database.select_samples_to_train_by_qbc(N))
                3) Bu sample'ları label'la (prep_labels)
                4) Modeli yeniden eğit (train_iterate)
                5) Stop condition kontrol et (check_stop_condition)
        """

        condition = 'unmet'
        labelled_so_far = 0
        max_labels = max_samples

        while condition == 'unmet':
            # 1) Komite tahminlerini çalıştır, DB'ye yaz
            CommitteeActiveLearning.committee_predict(max_samples=max_labels)

            # 2) En yüksek disagreement'li N örneği DB'den seç
            N = CommitteeActiveLearning.hyper_params["N"]

            # Bu fonksiyonu sen kendi database modülünde implement edeceksin.
            # Örn: SELECT ... ORDER BY disagreement_score DESC LIMIT N
            selected_samples = database.select_samples_to_train_by_qbc(N)

            if not selected_samples:
                print("No more samples to select (QBC).")
                break

            # 3) Label
            selected_samples_labelled = CommitteeActiveLearning.prep_labels(selected_samples)

            # 4) Train
            CommitteeActiveLearning.train_iterate(selected_samples_labelled)

            # 5) Label sayısını takip (isteğe bağlı)
            labelled_so_far += len(selected_samples_labelled)
            if max_labels is not None and labelled_so_far >= max_labels:
                print(f"Max labels reached: {max_labels}")
                break

            # 6) Stop condition kontrolü
            condition = CommitteeActiveLearning.check_stop_condition()

        print("Query by Committee active learning loop finished.")


if __name__ == "__main__":
    # Örnek kullanım:
    # Maksimum 50 sample label'layana kadar QBC loop çalıştır
    CommitteeActiveLearning.query_by_committee_sampling(max_samples=50)
