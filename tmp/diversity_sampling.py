import os
import sys
import csv
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from train import JobClassifierTrainer
from data_utils.database import database
import re

# ⚙️ CONFIG: Diversity sampling için max_samples değerini buradan değiştirin
MAX_SAMPLES = 5000

class ActiveLearning:
    hyper_params = {
        "N": 100,        # number of samples selected each iteration
        "I": 0.001,     # improvement threshold
    }

    @staticmethod
    def get_next_test_folder(base_path="./tests/diversity_sampling"):
        os.makedirs(base_path, exist_ok=True)
        existing = sorted([d for d in os.listdir(base_path) if d.startswith("test")])
        if not existing:
            return os.path.join(base_path, "test1")
        last_num = int(existing[-1][4:])
        return os.path.join(base_path, f"test{last_num + 1}")


    @staticmethod
    def get_next_result_file(folder):
        os.makedirs(folder, exist_ok=True)
        existing = [f for f in os.listdir(folder) if f.startswith("results") and f.endswith(".csv")]
        if not existing:
            return os.path.join(folder, "results_1.csv")
        
        nums = []
        for f in existing:
            m = re.search(r'results_(\d+)\.csv', f)
            if m:
                nums.append(int(m.group(1)))
        if not nums:
            return os.path.join(folder, "results_1.csv")
        last_num = max(nums)
        return os.path.join(folder, f"results_{last_num + 1}.csv")


    @staticmethod
    def prep_labels(samples):
        print("Please label your suggested samples... (simulated)")
        
        for sample in samples:
            sample_id = sample[0]
            label = sample[3]  # label zaten var (work_type)
            # DB'de güncelle
            database.update_labelled_sample(sample_id, label)

        return samples

    @staticmethod
    def train_iterate(samples, previous_trainer=None):
        trainer = JobClassifierTrainer()

        # 🔒 GARANTİ: tokenizer + model her zaman hazır
        if os.path.exists("./fine_tuned_eurobert"):
            trainer.load_model("./fine_tuned_eurobert")
        else:
            trainer.initialize_model()

        train_ds, eval_ds = trainer.prepare_datasets_from_tuples(
            samples,
            description_index=1,
            label_index=3
        )

        trained_trainer = trainer.train(train_ds, eval_ds)
        trainer.save_model("./fine_tuned_eurobert", trained_trainer)

        return trained_trainer

    @staticmethod
    def check_stop_condition(test_folder, previous_accuracy, new_accuracy):
        # İyileşme kontrolü
        if previous_accuracy is None:
            return False, None
        
        improvement = new_accuracy - previous_accuracy
        if improvement < ActiveLearning.hyper_params["I"]:
            return True, "min_improvement_not_met"
        
        # Havuz kontrolü
        unlabeled_count = database.get_unlabelled_count()
        if unlabeled_count == 0:
            return True, "pool_empty"
        
        return False, None

    @staticmethod
    def diversity_sampling(max_samples=None):
        # Test klasörünü otomatik oluştur
        test_folder = ActiveLearning.get_next_test_folder()
        os.makedirs(test_folder, exist_ok=True)

        iteration = 1
        previous_accuracy = None

        # Başlangıç için etiketli veri yoksa oluştur
        labeled_samples = database.get_labelled_samples()
        if not labeled_samples:
            print("Etiketli veri bulunamadı. Başlangıç için 100 örnek labeled olarak işaretleniyor...")
            database.initialize_labeled_pool(initial_size=100, random_seed=42)

        while True:
            # 1. Diversity sampling ile seç (max_samples parametresi ile embedding oluşturmayı sınırlandır)
            selected_samples = database.diversity_sampling_selection(
                N=ActiveLearning.hyper_params["N"],
                max_samples=max_samples
            )
            
            if not selected_samples:
                print("Seçilecek örnek kalmadı.")
                break
            
            # 2. Etiketle (label'lar zaten var, sadece is_labelled flag'ini güncelle)
            labeled_samples = ActiveLearning.prep_labels(selected_samples)

            # 3. Modeli eğit / güncelle
            trained_trainer = ActiveLearning.train_iterate(labeled_samples)

            # 4. Accuracy kontrolü
            new_accuracy = trained_trainer.evaluate().get("eval_accuracy", None)

            # 5. Stop condition
            stop, reason = ActiveLearning.check_stop_condition(test_folder, previous_accuracy, new_accuracy)

            # 6. Sonuçları CSV'ye yaz
            result_file = ActiveLearning.get_next_result_file(test_folder)
            with open(result_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "iteration", "n_labeled", "previous_accuracy", "new_accuracy", "stop_condition", "stop_reason"
                ])
                writer.writeheader()
                writer.writerow({
                    "iteration": iteration,
                    "n_labeled": len(labeled_samples),
                    "previous_accuracy": previous_accuracy,
                    "new_accuracy": new_accuracy,
                    "stop_condition": stop,
                    "stop_reason": reason
                })

            print(f"Iteration {iteration} results saved to {result_file}")
            iteration += 1
            previous_accuracy = new_accuracy

            if stop:
                print(f"Stopping: {reason}")
                break

if __name__ == '__main__':
    ActiveLearning.diversity_sampling(max_samples=MAX_SAMPLES)

