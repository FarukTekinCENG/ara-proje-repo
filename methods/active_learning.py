import os
import sys
import csv
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from train import JobClassifierTrainer
from model import ModelPredictor
from data_utils.database import database
import re

class ActiveLearning:
    hyper_params = {
        "N": 5,         # number of samples selected each iteration
        "I": 0.001,     # improvement threshold
        "T": 0.3        # model prediction certainty threshold
    }

    @staticmethod
    def get_next_test_folder(base_path="./tests"):
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
    def model_predict(max_samples=None):
        predictor = ModelPredictor("./fine_tuned_eurobert")

        batch_size = 1000
        total_processed = 0
        page = 0

        while True:
            if max_samples is not None and total_processed >= max_samples:
                print(f"Max samples limit reached: {max_samples}")
                break

            offset = page * batch_size
            batch = database.get_unlabelled_samples(batch_size, offset)
            if not batch:
                break

            if max_samples is not None:
                remaining = max_samples - total_processed
                if len(batch) > remaining:
                    batch = batch[:remaining]

            print(f"Page {page}: {len(batch)} records")
            total_processed += len(batch)
            page += 1

            for x in batch:
                result = predictor.predict(x[1])
                uncertainty = 1 - result["confidence"]
                database.save_model_prediction(
                    sample_id=x[0],
                    predicted_class=result["predicted_class"],
                    uncertainty_score=uncertainty
                )

    @staticmethod
    def prep_labels(samples):
        print("Please label your suggested samples... (simulated)")

        for sample in samples:
            sample_id = sample[0]
            predicted_label = sample[3]  # ya da simüle edilmiş label
            # DB’de güncelle
            database.update_labelled_sample(sample_id, predicted_label)

        return samples

    @staticmethod
    def train_iterate(samples, previous_trainer=None):
        trainer = JobClassifierTrainer()

        # tokenizer + model her zaman hazır
        if os.path.exists("./fine_tuned_eurobert"):
            trainer.load_model("./fine_tuned_eurobert")
        else:
            trainer.initialize_model()

        # Use the incoming samples only for training (no internal split)
        train_ds, eval_ds = trainer.prepare_datasets_from_tuples(
            samples,
            description_index=1,
            label_index=3,
            split=False,
        )

        trained_trainer = trainer.train(train_ds, None)
        trainer.save_model("./fine_tuned_eurobert", trained_trainer)

        # Return both the JobClassifierTrainer instance (holds tokenizer/label encoder/model)
        # and the underlying Trainer object
        return trainer, trained_trainer

    @staticmethod
    def check_stop_condition(test_folder):
        # Ortalama uncertainty üzerinden T threshold kontrolü
        pool_scores = database.get_all_uncertainty_scores()
        if not pool_scores:
            return True, "pool_empty"
        # Stringleri float'a çevir
        pool_scores = [float(x) for x in pool_scores]
        avg_score = sum(pool_scores)/len(pool_scores)
        if avg_score < ActiveLearning.hyper_params["T"]:
            return True, "T_threshold_reached"
        return False, None


    @staticmethod
    def uncertainty_sampling():
        return database.uncertainty_sampling_selection(ActiveLearning.hyper_params["N"])

    @staticmethod
    def diversity_sampling():
        return database.diversity_sampling_selection(ActiveLearning.hyper_params["N"])

    @staticmethod
    def run(function_algorithm, max_samples=None, test_samples=None):
        # Test klasörünü otomatik oluştur
        test_folder = ActiveLearning.get_next_test_folder()
        os.makedirs(test_folder, exist_ok=True)

        iteration = 1
        previous_accuracy = None

        while True:
            # 1. Model tahmini
            ActiveLearning.model_predict(max_samples)

            # 2. Uncertainty sampling ile seç
            selected_samples = function_algorithm()

            # ara adim: veriyi etiketle
            labeled_samples = ActiveLearning.prep_labels(selected_samples)

            # 3. Modeli eğit / güncelle
            trainer_obj, trained_trainer = ActiveLearning.train_iterate(labeled_samples)

            # 4. Accuracy kontrolü
            new_accuracy = None

            # If an external test set is provided, evaluate on it using the trained trainer object
            if test_samples:
                # Build test dataset using trainer's label encoder and tokenizer
                descriptions = []
                labels = []
                for row in test_samples:
                    if len(row) > 3 and row[1] and row[3] is not None:
                        descriptions.append(str(row[1]))
                        labels.append(str(row[3]))

                if trainer_obj.label_encoder is not None and labels:
                    try:
                        encoded_labels = trainer_obj.label_encoder.transform(labels)
                        from datasets import Dataset
                        test_ds = Dataset.from_dict({
                            "description": descriptions,
                            "label": encoded_labels,
                        })
                        test_ds = test_ds.map(trainer_obj.tokenize_function, batched=True)
                        test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

                        metrics = trainer_obj.evaluate(test_ds)
                        new_accuracy = metrics.get("eval_accuracy", None)
                    except Exception:
                        new_accuracy = None
                else:
                    new_accuracy = None

            # 5. Stop condition
            stop, reason = ActiveLearning.check_stop_condition(test_folder)

            # 6. Sonuçları CSV’ye yaz
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
    ActiveLearning.run(ActiveLearning.uncertainty_sampling, max_samples=50)

