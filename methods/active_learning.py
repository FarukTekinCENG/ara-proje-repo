import os
import sys
import csv
import torch
from transformers import TrainingArguments, Trainer
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from train import JobClassifierTrainer
from model import ModelPredictor
from data_utils.database import database
import re
import shutil

# ⚙️ CONFIGURABLE PARAMETERS
MAX_SAMPLES = 100
TEST_SAMPLE_LIMIT = 200
BASE_TRAIN_SIZE = 10

class ActiveLearning:
    hyper_params = {
        "N": 100,         # number of samples selected each iteration
        "I": 0.001,     # improvement threshold
        "T": 0.001,        # model prediction un-certainty threshold
        "max_iterations": 20,  # maximum number of iterations
        "succcess_rate_threshold": 0.85,  # desired accuracy to stop
    }
    BASE_DIR = "./base_classifier"
    RUNS_BASE = "./tests"
    TRAINED_BASE = "./trained_models"
    # Committee models (HF hub ids or local paths). Update to use desired models.
    model_list = [
        "FacebookAI/roberta-base",
        "google-bert/bert-base-cased",
    ]

    @staticmethod
    def get_next_model_folder(base_path=None):
        base_path = base_path or ActiveLearning.TRAINED_BASE
        os.makedirs(base_path, exist_ok=True)
        existing = sorted([d for d in os.listdir(base_path) if d.startswith("model")])
        if not existing:
            return os.path.join(base_path, "model1")
        last_num = int(existing[-1][5:])
        return os.path.join(base_path, f"model{last_num + 1}")

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
    def model_predict(max_samples=None, model_dir=None):
        # Predict with the provided model_dir, falling back to BASE_DIR
        model_dir = model_dir or ActiveLearning.BASE_DIR
        predictor = ModelPredictor(model_dir)

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
    def initialize_and_train_base_classifier(
        train_size=500,
        base_dir=None,
        source="db",
    ):
        """
        Initialize AND train base classifier with MINIMAL task adaptation.
        Encoder is frozen. Only classification head is trained.
        """

        base_dir = base_dir or ActiveLearning.BASE_DIR
        os.makedirs(base_dir, exist_ok=True)

        print(f"Initializing base classifier with {train_size} samples")

        # 1. Load labeled samples
        if source == "db":
            samples = database.get_labeled_samples(train_size)
        else:
            raise ValueError("Only DB source supported for base init")

        if not samples:
            raise RuntimeError("No labeled samples found for base classifier initialization")

        # 2. Create trainer
        trainer = JobClassifierTrainer()

        # 3. Initialize model (fresh)
        trainer.initialize_model()

        # 🔒 🔴 KRİTİK ADIM: ENCODER'I DONDUR
        ActiveLearning.freeze_encoder(trainer.model)

        # 4. Prepare dataset
        train_ds, _ = trainer.prepare_datasets_from_tuples(
            samples,
            description_index=1,
            label_index=3,
            split=False,
        )

        print(f"Base classifier training on {len(train_ds)} samples")

        # 5. Train — ⚠️ SADECE 1 EPOCH
        transformers_trainer = trainer.train(
            train_ds,
            None,
            num_train_epochs=1,      # ⬅️ MIN
            learning_rate=5e-5,      # ⬅️ head için biraz yüksek
        )

        # 6. Save
        #trainer.save_model(base_dir, transformers_trainer)
        trainer.save_model(base_dir)

        print(f"Base classifier trained (encoder frozen) and saved to {base_dir}")

        return trainer

    @staticmethod
    def freeze_encoder(model):
        """
        Freeze all encoder parameters, keep classification head trainable.
        """
        # HF modellerinde encoder genelde base_model veya bert/roberta altında olur
        if hasattr(model, "base_model"):
            encoder = model.base_model
        elif hasattr(model, "bert"):
            encoder = model.bert
        elif hasattr(model, "roberta"):
            encoder = model.roberta
        else:
            raise RuntimeError("Encoder not found in model")

        for param in encoder.parameters():
            param.requires_grad = False

        print("🔒 Encoder frozen (classification head only)")

    @staticmethod
    def train_iterate(samples, source_model_dir=None, save_dir=None, previous_trainer=None):
        trainer = JobClassifierTrainer()
        
        # Load tokenizer/model from given source_model_dir (or base)
        source_model_dir = source_model_dir or ActiveLearning.BASE_DIR
        if os.path.exists(source_model_dir):
            trainer.load_model(source_model_dir)
        else:
            trainer.initialize_model()
        
        # DEBUG: Check what we have in samples
        if samples and len(samples) > 0:
            print(f"DEBUG: Number of samples for training: {len(samples)}")
            print(f"DEBUG: First sample label at index 3: {samples[0][3] if len(samples[0]) > 3 else 'No label'}")
        
        try:
            # FIXED: Eski train.py ile uyumlu hale getir - split parametresi var
            # Active learning için tüm veriyi kullanıyoruz, test split yok
            train_ds, eval_ds = trainer.prepare_datasets_from_tuples(
                samples,
                description_index=1,
                label_index=3,
                split=False,  # Tüm veriyi eğitim için kullan
            )
            print(f"DEBUG: Dataset prepared successfully. Train dataset size: {len(train_ds)}")
            
        except Exception as e:
            print(f"ERROR in prepare_datasets_from_tuples: {e}")
            if samples and len(samples) > 0:
                print(f"Sample 0 content: {samples[0]}")
                print(f"Sample 0 length: {len(samples[0])}")
            raise
        
        # Train with train_ds only (eval_ds will be None)
        transformers_trainer = trainer.train(train_ds, None)
        
        # Save into provided save_dir (create if missing)
        save_dir = save_dir or "./fine_tuned_eurobert"
        os.makedirs(save_dir, exist_ok=True)
        #trainer.save_model(save_dir, transformers_trainer)
        trainer.save_model(save_dir)

        # Return both the custom trainer and the transformers trainer
        return trainer, transformers_trainer
    
    @staticmethod
    def evaluate_on_test_set(trainer_obj, test_samples):
        """Evaluate model on test samples and return accuracy"""
        if not test_samples:
            return None
        
        try:
            # Test dataset hazırla (split=True yapmıyoruz, direkt tüm test verisi)
            test_ds, _ = trainer_obj.prepare_datasets_from_tuples(
                test_samples,
                description_index=1,
                label_index=3,
                split=False,  # Test verisi için split yok
            )
            
            # Create a new Trainer for evaluation
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            trainer_obj.model.to(device)
            
            # Training arguments ile Trainer oluştur
            training_args = TrainingArguments(
                output_dir="./temp_eval",
                eval_strategy="no",
                per_device_eval_batch_size=16,
                remove_unused_columns=False,
                report_to="none",
                fp16=True,
            )
            
            eval_trainer = Trainer(
                model=trainer_obj.model,
                args=training_args,
                tokenizer=trainer_obj.tokenizer,
                compute_metrics=trainer_obj.compute_metrics,
            )
            
            # Evaluate
            eval_result = eval_trainer.evaluate(test_ds)
            return eval_result.get("eval_accuracy", None)
            
        except Exception as e:
            print(f"Warning: Evaluation failed: {e}")
            # Try simple evaluation as fallback
            try:
                return ActiveLearning.simple_evaluate(trainer_obj, test_samples)
            except Exception as e2:
                print(f"Simple evaluation also failed: {e2}")
                return None
    
    @staticmethod
    def simple_evaluate(trainer_obj, test_samples):
        """Basit accuracy hesaplama (fallback method)"""
        if not trainer_obj.label_encoder:
            return None
        
        correct = 0
        total = 0
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trainer_obj.model.to(device)
        trainer_obj.model.eval()
        
        with torch.no_grad():
            for sample in test_samples:
                if len(sample) < 4:
                    continue
                    
                desc = sample[1]
                true_label = sample[3]
                
                if not desc or not true_label:
                    continue
                
                # Tokenize
                inputs = trainer_obj.tokenizer(
                    str(desc),
                    truncation=True,
                    padding=True,
                    max_length=256,
                    return_tensors="pt"
                ).to(device)
                
                # Predict
                outputs = trainer_obj.model(**inputs)
                pred_id = torch.argmax(outputs.logits, dim=-1).item()
                
                # Label encoder ile decode et
                try:
                    # Label encoder'ın sınıflarını al
                    classes = trainer_obj.label_encoder.classes_
                    if pred_id < len(classes):
                        pred_label = classes[pred_id]
                        if str(pred_label) == str(true_label):
                            correct += 1
                    total += 1
                except:
                    continue
        
        return correct / total if total > 0 else None
        
    @staticmethod
    def check_stop_condition(test_folder, iteration, previous_accuracy=None, new_accuracy=None):
        # Stop condition checks using multiple criteria:
        # - avg uncertainty T threshold
        # - maximum iterations
        # - success rate threshold on external test set accuracy
        # - minimal improvement I
        pool_scores = database.get_all_uncertainty_scores()
        if not pool_scores:
            return True, "pool_empty"

        # convert to floats and compute average uncertainty
        pool_scores = [float(x) for x in pool_scores]
        avg_score = sum(pool_scores) / len(pool_scores)
        if avg_score < ActiveLearning.hyper_params.get("T", 0.0):
            return True, "T_threshold_reached"

        # max iterations
        max_it = ActiveLearning.hyper_params.get("max_iterations")
        if max_it is not None and iteration is not None and iteration >= max_it:
            return True, "max_iterations_reached"

        # success rate threshold based on external test accuracy
        sr = ActiveLearning.hyper_params.get("succcess_rate_threshold")
        if new_accuracy is not None and sr is not None and new_accuracy >= sr:
            return True, "success_rate_reached"

        # improvement threshold I
        I = ActiveLearning.hyper_params.get("I")
        if previous_accuracy is not None and new_accuracy is not None and I is not None:
            try:
                diff=(new_accuracy - previous_accuracy)
                if diff > 0 and diff < I:
                    return True, "improvement_below_I"
            except Exception:
                pass

        return False, None

    @staticmethod
    def random_sampling():
        N = ActiveLearning.hyper_params.get("N", 5)
        query = """
            SELECT id, description, is_labelled, label, model_prediction, uncertainty_score
            FROM pool
                        WHERE is_labelled = 'FALSE'
                            AND description IS NOT NULL
            ORDER BY RANDOM()
            LIMIT %s;
        """
        with database.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (N,))
                return cur.fetchall()

    @staticmethod
    def uncertainty_sampling():
        return database.uncertainty_sampling_selection(ActiveLearning.hyper_params["N"])

    @staticmethod
    def diversity_sampling():
        return database.diversity_sampling_selection(ActiveLearning.hyper_params["N"])

    @staticmethod
    def query_by_comitee(max_samples=MAX_SAMPLES):
        import collections
        from collections import defaultdict
        import math

        N = ActiveLearning.hyper_params.get("N", 5)
        models = ActiveLearning.model_list
        created_tables = []

        try:
            # 1️⃣ Create temp tables per member
            for idx, m in enumerate(models):
                tbl = f"committee_{idx}"
                created_tables.append(tbl)
                with database.get_db_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(f"""
                            CREATE TABLE IF NOT EXISTS {tbl} (
                                pool_id INTEGER PRIMARY KEY,
                                model_prediction TEXT,
                                uncertainty_score DOUBLE PRECISION
                            );
                        """)
                    conn.commit()

            # 2️⃣ Load each model & make predictions
            for idx, model_name in enumerate(models):
                predictor = ModelPredictor(model_name)
                try:
                    predictor.load_model()
                    print(f"[committee] member {idx} loaded: {model_name}")
                except Exception as e:
                    print(f"[committee] failed to load member {idx} ({model_name}): {e}")
                    continue

                batch_size = 1000
                page = 0
                total_processed = 0
                rows_to_insert = []

                while True:
                    batch = database.get_unlabelled_samples(batch_size, page * batch_size)
                    if not batch or (max_samples is not None and total_processed >= max_samples):
                        break

                    # max_samples kontrolü
                    if max_samples is not None:
                        remaining = max_samples - total_processed
                        if len(batch) > remaining:
                            batch = batch[:remaining]

                    for x in batch:
                        pool_id, desc = x[0], x[1]
                        if not desc:
                            continue
                        try:
                            res = predictor.predict(text=desc)
                            pred = res.get("predicted_class")
                            probs = res.get("probs")
                            if probs:
                                ent = -sum([p * math.log(p + 1e-12) for p in probs])
                                unc = float(ent)
                            else:
                                unc = 1 - res.get("confidence", 1.0)
                        except Exception as e:
                            print(f"[committee] prediction error member {idx} id {pool_id}: {e}")
                            pred = None
                            unc = None
                        rows_to_insert.append((pool_id, pred, unc))

                    # Insert batch
                    if rows_to_insert:
                        with database.get_db_connection() as conn:
                            with conn.cursor() as cur:
                                cur.executemany(
                                    f"INSERT INTO committee_{idx} (pool_id, model_prediction, uncertainty_score) "
                                    f"VALUES (%s, %s, %s) "
                                    f"ON CONFLICT (pool_id) DO UPDATE SET model_prediction = EXCLUDED.model_prediction, uncertainty_score = EXCLUDED.uncertainty_score;",
                                    rows_to_insert
                                )
                            conn.commit()
                        rows_to_insert = []

                    total_processed += len(batch)
                    page += 1

            # 3️⃣ Aggregate member predictions and compute disagreement
            preds_by_id = defaultdict(list)
            uncert_by_id = defaultdict(list)
            for idx in range(len(models)):
                table = f"committee_{idx}"
                with database.get_db_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(f"SELECT pool_id, model_prediction, uncertainty_score FROM {table};")
                        for pid, mpred, unc in cur.fetchall():
                            preds_by_id[pid].append(mpred)
                            if unc is not None:
                                uncert_by_id[pid].append(float(unc))

            # 4️⃣ Compute disagreement score and select top-N
            scored = []
            for pid, preds in preds_by_id.items():
                distinct = len([p for p in set(preds) if p is not None])
                avg_unc = sum(uncert_by_id.get(pid, [0])) / max(1, len(uncert_by_id.get(pid, [])))
                scored.append((pid, (distinct, avg_unc)))

            scored.sort(key=lambda x: (x[1][0], x[1][1]), reverse=True)
            selected_ids = [s[0] for s in scored[:N]]

            if not selected_ids:
                return []

            # 5️⃣ Fetch full sample rows
            placeholders = ','.join(['%s'] * len(selected_ids))
            query = f"""
                SELECT id, description, is_labelled, label, model_prediction, uncertainty_score
                FROM pool
                WHERE id IN ({placeholders})
                ORDER BY id;
            """
            with database.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, tuple(selected_ids))
                    rows = cur.fetchall()

            return rows

        finally:
            # 6️⃣ Cleanup temp tables
            for tbl in created_tables:
                try:
                    with database.get_db_connection() as conn:
                        with conn.cursor() as cur:
                            cur.execute(f"DROP TABLE IF EXISTS {tbl};")
                        conn.commit()
                except Exception:
                    pass

    @staticmethod
    def run(function_algorithm, max_samples=None, test_samples=None, test_sample_limit=TEST_SAMPLE_LIMIT, test_from_db=True, base_train_size=BASE_TRAIN_SIZE):
        # tum set
        all_labeled_samples = []

        # Ensure base classifier exists
        os.makedirs(ActiveLearning.RUNS_BASE, exist_ok=True)
        if not os.path.exists(ActiveLearning.BASE_DIR):
            print("Base classifier not found. Initializing and training...")

            ActiveLearning.initialize_and_train_base_classifier(
                train_size=base_train_size,          # 🔧 PARAMETRE
                base_dir=ActiveLearning.BASE_DIR,
            )

        # Ensure committee member local paths exist: copy base classifier into missing local dirs
        for m in ActiveLearning.model_list:
            if not isinstance(m, str):
                continue
            if m == ActiveLearning.BASE_DIR:
                continue
            # if path already exists, skip
            if os.path.exists(m):
                continue
            # Heuristic: treat as local path if it starts with '.' or '/' or refers to trained_models
            if m.startswith('.') or m.startswith('/') or m.startswith(ActiveLearning.TRAINED_BASE) or m.startswith('./') or m.startswith('trained_models'):
                try:
                    os.makedirs(os.path.dirname(m), exist_ok=True)
                    shutil.copytree(ActiveLearning.BASE_DIR, m)
                    print(f"Initialized committee member from base classifier: {m}")
                except Exception as e:
                    print(f"Warning: could not initialize committee member {m}: {e}")
            else:
                # assume a HF hub model name (skip local initialization)
                pass

        # Reset pool for a fresh run (clear labels and model predictions)
        try:
            reset_res = database.reset_pool(clear_labels=False, clear_predictions=True)
            print(f"Pool reset: {reset_res}")
        except Exception as e:
            print(f"Warning: failed to reset pool: {e}")

        # Create a single test run folder for this run
        test_folder = ActiveLearning.get_next_test_folder(base_path=ActiveLearning.RUNS_BASE)
        os.makedirs(test_folder, exist_ok=True)

        # Create a model folder for this run under trained_models (one per run)
        run_model_dir = ActiveLearning.get_next_model_folder()
        os.makedirs(run_model_dir, exist_ok=True)

        # Set current model dir to base initially; subsequent iterations will use models saved in run_model_dir
        current_model_dir = ActiveLearning.BASE_DIR

        # determine test id for this run (numeric incremental)
        try:
            test_id_num = database.get_next_test_id()
        except Exception:
            test_id_num = 1

        test_id = str(test_id_num)

        iteration = 1
        previous_accuracy = None
        # Record active-learning method name and DATA_SIZE
        method_name = function_algorithm.__name__ if callable(function_algorithm) else str(function_algorithm)
        data_size = max_samples  # DATA_SIZE == max_samples per user instruction

        # Eğer diversity_sampling ise, max_samples parametresini geçirmek için closure ile sarmala
        if callable(function_algorithm) and function_algorithm.__name__ == "diversity_sampling":
            function_algorithm = lambda: database.diversity_sampling_selection(
                N=ActiveLearning.hyper_params["N"],
                max_samples=max_samples
            )

        # If requested, load fixed test set from DB once (will be used for all iterations)
        if test_from_db:
            try:
                test_samples = database.get_test_samples(test_sample_limit)
                print(f"Loaded {len(test_samples)} test samples from DB")
            except Exception as e:
                print(f"Warning: failed to load test samples from DB: {e}")
                test_samples = None

        # ============================================================
        # BASE CLASSIFIER EVALUATION (iteration = 0)
        # ============================================================
        base_accuracy = None

        if test_samples:
            try:
                print(f"Evaluating BASE classifier on test set ({len(test_samples)} samples)...")

                base_trainer = JobClassifierTrainer()
                base_trainer.load_model(ActiveLearning.BASE_DIR)

                base_accuracy = ActiveLearning.evaluate_on_test_set(
                    base_trainer, test_samples
                )

                print(f"BASE test accuracy: {base_accuracy}")

            except Exception as e:
                print(f"Warning: BASE evaluation failed: {e}")
                base_accuracy = None

        # --- CSV write (same schema as others) ---
        base_result_file = ActiveLearning.get_next_result_file(test_folder)
        with open(base_result_file, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "iteration",
                    "n_labeled",
                    "previous_accuracy",
                    "new_accuracy",
                    "stop_condition",
                    "stop_reason",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "iteration": 0,
                    "n_labeled": 0,
                    "previous_accuracy": None,
                    "new_accuracy": base_accuracy,
                    "stop_condition": False,
                    "stop_reason": "base_classifier",
                }
            )

        # --- Remote DB insert (aynı muamele) ---
        try:
            scores = database.get_all_uncertainty_scores()
            avg_uncertainty = sum([float(x) for x in scores]) / len(scores) if scores else None
            metrics = {"accuracy": base_accuracy, "avg_uncertainty": avg_uncertainty} if base_accuracy is not None else {"avg_uncertainty": avg_uncertainty}
            params = {
                "run_model_dir": ActiveLearning.BASE_DIR,
                "test_folder": test_folder,
                "method": "BASE",
                "DATA_SIZE": data_size,
            }

            database.insert_test_result_remote(
                test_id=test_id,
                iteration_no=0,
                model_name="base_classifier",
                train_data_size=0,
                method="BASE",
                data_size=data_size,
                N=None,
                T=None,
                I=None,
                metrics=metrics,
                params=params,
                run_by=os.getenv("USER") or os.getenv("USERNAME"),
                notes="base_classifier",
            )

            print("Inserted BASE classifier result to remote DB")

        except Exception as e:
            print(f"Warning: failed to insert BASE result to DB: {e}")

        # Base accuracy artık referans noktası
        previous_accuracy = base_accuracy

        while True:
            print(f"\n--- Iteration {iteration} ---")
            
            # 1. Model tahmini (use current_model_dir)
            ActiveLearning.model_predict(max_samples, model_dir=current_model_dir)

            # 2. Algoritma ile etiketlenecek ornekleri sec
            selected_samples = function_algorithm()
            print(f"Selected {len(selected_samples)} samples for labeling")

            # ara adim: veriyi etiketle
            labeled_samples = ActiveLearning.prep_labels(selected_samples)
            all_labeled_samples.extend(labeled_samples)
            print(f"Total labeled samples so far: {len(all_labeled_samples)}")

            # 3. Modeli eğit / güncelle
            # Train and save into the run-specific model folder (overwrite each iteration)
            try:
                trainer_obj, transformers_trainer = ActiveLearning.train_iterate(
                    all_labeled_samples, #labeled_samples,
                    source_model_dir=current_model_dir,
                    save_dir=run_model_dir,
                )
                print(f"Model trained successfully on {len(labeled_samples)} samples")
            except Exception as e:
                print(f"ERROR in train_iterate: {e}")
                # If training fails, break the loop
                break

            # After training, switch prediction to the latest model in run_model_dir
            current_model_dir = run_model_dir

            # 4. Accuracy kontrolü
            new_accuracy = None

            # If an external test set is provided, evaluate on it
            if test_samples:
                try:
                    print(f"Evaluating on test set ({len(test_samples)} samples)...")
                    new_accuracy = ActiveLearning.evaluate_on_test_set(trainer_obj, test_samples)
                    print(f"Test accuracy: {new_accuracy}")
                except Exception as e:
                    print(f"Warning: Evaluation failed: {e}")
                    new_accuracy = None

            # 5. Stop condition (use iteration and accuracy if available)
            stop, reason = ActiveLearning.check_stop_condition(test_folder, iteration, previous_accuracy, new_accuracy)

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

            # 7. Save result to Postgres: prefer remote (Neon) for results, fallback to local
            try:
                scores = database.get_all_uncertainty_scores()
                avg_uncertainty = sum([float(x) for x in scores]) / len(scores) if scores else None
                metrics = {"accuracy": new_accuracy, "avg_uncertainty": avg_uncertainty} if base_accuracy is not None else {"avg_uncertainty": avg_uncertainty}
                params = {
                    "run_model_dir": run_model_dir,
                    "test_folder": test_folder,
                    "previous_accuracy": previous_accuracy,
                    "method": method_name,
                    "DATA_SIZE": data_size,
                }

                try:
                    # Try remote insert (requires NEON_API_KEY or DATABASE_URL in env)
                    inserted_id = database.insert_test_result_remote(
                        test_id=test_id,
                        iteration_no=iteration,
                        model_name=getattr(trainer_obj, "model_name", None) or "unknown",
                        train_data_size=len(labeled_samples) if labeled_samples else 0,
                        method=method_name,
                        data_size=data_size,
                        N=ActiveLearning.hyper_params.get("N"),
                        T=ActiveLearning.hyper_params.get("T"),
                        I=ActiveLearning.hyper_params.get("I"),
                        metrics=metrics,
                        params=params,
                        run_by=os.getenv("USER") or os.getenv("USERNAME"),
                        notes=reason,
                    )
                    print(f"Inserted remote result row id={inserted_id} for test_id={test_id} iteration={iteration}")
                except RuntimeError as re:
                    # Remote not configured — fall back to local insert
                    print(f"Remote DB not configured: {re}. Falling back to local DB.")
                    inserted_id = database.insert_test_result(
                        test_id=test_id,
                        iteration_no=iteration,
                        model_name=getattr(trainer_obj, "model_name", None) or "unknown",
                        train_data_size=len(labeled_samples) if labeled_samples else 0,
                        method=method_name,
                        data_size=data_size,
                        N=ActiveLearning.hyper_params.get("N"),
                        T=ActiveLearning.hyper_params.get("T"),
                        I=ActiveLearning.hyper_params.get("I"),
                        metrics=metrics,
                        params=params,
                        run_by=os.getenv("USER") or os.getenv("USERNAME"),
                        notes=reason,
                    )
                    print(f"Inserted local result row id={inserted_id} for test_id={test_id} iteration={iteration}")
            except Exception as e:
                print(f"Warning: failed to insert result row to DB: {e}")

            print(f"Iteration {iteration} results saved to {result_file}")
            iteration += 1
            previous_accuracy = new_accuracy

            if stop:
                print(f"Stopping: {reason}")
                break

            # Safety check: don't exceed maximum iterations
            if iteration > ActiveLearning.hyper_params["max_iterations"]:
                print(f"Reached maximum iterations ({ActiveLearning.hyper_params['max_iterations']})")
                break

if __name__ == '__main__':
    methods = [
        #ActiveLearning.random_sampling,
        ActiveLearning.uncertainty_sampling,
        ActiveLearning.diversity_sampling,
        ActiveLearning.query_by_comitee,
    ]

    for m in methods:
        print(f"\n=== Running method: {m.__name__} ===")
        try:
            ActiveLearning.run(m, max_samples=MAX_SAMPLES, test_from_db=True)
        except Exception as e:
            print(f"Error running {m.__name__}: {e}")