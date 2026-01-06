import os
import sys
import csv
import torch
import random
import shutil
import json
from transformers import TrainingArguments, Trainer
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from train import JobClassifierTrainer
from model import ModelPredictor
from utils.database_csv import database  # Database yerine RAMDatabase
from utils.metrics import compute_minority_metrics
import re
import shutil

try:
    print(f"[dataset] database.data_csv_path={getattr(database, 'data_csv_path', None)}")
    print(f"[dataset] pool_size={len(getattr(database, 'pool', []) or [])} test_size={len(getattr(database, 'test_data', []) or [])}")
except Exception as e:
    print(f"Warning: failed to print dataset info: {e}")

# ⚙️ CONFIGURABLE PARAMETERS
MAX_SAMPLES = 18000
TEST_SAMPLE_LIMIT = 2000
BASE_TRAIN_SIZE = 100

class ActiveLearning:
    hyper_params = {
        "N": 2000,         # number of samples selected each iteration
        "T": 0.01,        # model prediction un-certainty threshold
        "T_patience": 3,  # stop if mean_uncertainty(selected_batch) < T for this many consecutive iterations
        "uncertainty_plateau_eps": 0.0,   # disable uncertainty plateau (set to 0)
        "uncertainty_plateau_patience": 0,  # disable uncertainty plateau (set to 0)
        "accuracy_threshold": None,  # optional: stop if test accuracy >= threshold (None disables)
        "label_budget": 10000, # None,  # optional hard budget on total labelled samples used for training
        "max_iterations": 12,  # maximum number of iterations
        "stratified_batch": False,  # make per-iteration selected batch roughly class-balanced
        "seed": None,
        "deterministic": False,
        "predict_batch_size": 5000,
        "verbose_pages": False,
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
    def normalize_model_name(model_name):
        if model_name is None:
            return "unknown"
        s = str(model_name)
        if not s:
            return "unknown"
        if " (" in s and s.endswith(")"):
            return s.split(" (", 1)[0]
        return s

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

        batch_size = ActiveLearning.hyper_params.get("predict_batch_size", 1000)
        if not isinstance(batch_size, int) or batch_size <= 0:
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

            if ActiveLearning.hyper_params.get("verbose_pages", False):
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
            # Gerçek etiketi kullan (zaten var)
            true_label = sample[3]  # label index
            if true_label:
                database.update_labelled_sample(sample_id, true_label)

        return samples

    @staticmethod
    def fallback_random_unlabelled(n: int):
        try:
            candidates = [r for r in database.pool if r[2] == 'FALSE' and r[1] and r[3] is not None]
            if not candidates:
                return []
            k = min(int(n), len(candidates))
            return random.sample(candidates, k)
        except Exception:
            return []

    @staticmethod
    def initialize_and_train_base_classifier(
        train_size=500,
        base_dir=None,
        source="db",
        seed=None,
    ):
        """
        Initialize AND train base classifier with MINIMAL task adaptation.
        Encoder is frozen. Only classification head is trained.
        """

        base_dir = base_dir or ActiveLearning.BASE_DIR
        os.makedirs(base_dir, exist_ok=True)

        print(f"Initializing base classifier with {train_size} samples")

        # Seed: mark a small subset of pool as labelled so we can train a base model.
        # In the original design, the whole pool starts FALSE; samples become TRUE only when used for training.
        try:
            if seed is not None:
                random.seed(seed)
            seed_candidates = [r for r in database.pool if r[2] == 'FALSE' and r[3] is not None and r[1]]
            seed_n = min(train_size, len(seed_candidates))
            if seed_n <= 0:
                raise RuntimeError("No candidates available to seed base classifier")
            seed_samples = random.sample(seed_candidates, seed_n)
            for s in seed_samples:
                database.update_labelled_sample(s[0], s[3])
            print(f"Seeded base training set: marked {seed_n} samples as labelled")
        except Exception as e:
            print(f"Warning: failed to seed base labelled samples: {e}")

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
        
        # Persist only the base HF model id as model_name; run folder is already stored via params['run_model_dir']
        try:
            base_id = getattr(JobClassifierTrainer, "model_name", None) or getattr(trainer, "model_name", None) or "unknown"
        except Exception:
            base_id = "unknown"
        trainer.model_name = str(base_id)

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
    def compute_avg_uncertainty_on_samples(trainer_obj, samples):
        if not samples:
            return None

        try:
            ds, _ = trainer_obj.prepare_datasets_from_tuples(
                samples,
                description_index=1,
                label_index=3,
                split=False,
            )

            hf_trainer = getattr(trainer_obj, "trainer", None)
            if hf_trainer is None:
                training_args = TrainingArguments(
                    output_dir="./temp_eval",
                    eval_strategy="no",
                    per_device_eval_batch_size=16,
                    remove_unused_columns=False,
                    report_to="none",
                    fp16=torch.cuda.is_available(),
                )
                hf_trainer = Trainer(
                    model=trainer_obj.model,
                    args=training_args,
                    tokenizer=trainer_obj.tokenizer,
                )

            preds = hf_trainer.predict(ds)
            logits = preds.predictions
            if logits is None:
                return None

            logits_t = torch.tensor(logits)
            probs = torch.softmax(logits_t, dim=-1)
            confidences = probs.max(dim=-1).values
            uncertainties = 1.0 - confidences
            return float(uncertainties.mean().item())

        except Exception as e:
            print(f"Warning: failed to compute avg_uncertainty on samples: {e}")
            return None

    @staticmethod
    def compute_mean_uncertainty_for_selected_samples(samples, model_dir=None):
        if not samples:
            return None

        # Prefer already-computed uncertainty_score in sample tuples if present.
        scores = []
        for s in samples:
            try:
                if len(s) >= 6 and s[5] is not None:
                    scores.append(float(s[5]))
            except Exception:
                pass

        if scores:
            return sum(scores) / len(scores)

        # Fallback: compute uncertainty on the fly from the current model.
        try:
            model_dir = model_dir or ActiveLearning.BASE_DIR
            predictor = ModelPredictor(model_dir)
            for s in samples:
                desc = s[1] if len(s) > 1 else None
                if not desc:
                    continue
                result = predictor.predict(desc)
                uncertainty = 1 - result["confidence"]
                scores.append(float(uncertainty))
            return sum(scores) / len(scores) if scores else None
        except Exception as e:
            print(f"Warning: failed to compute mean uncertainty for selected batch: {e}")
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
        """Stop condition checks using multiple criteria.

        In-memory variant uses only:
        - budget (label_budget / max_iterations)
        - plateau on selected batch mean uncertainty (eps + patience)

        Args:
            test_folder: kept for API compatibility (unused)
            iteration: current iteration counter
            previous_accuracy/new_accuracy: kept for API compatibility (unused)
        """
        # Backward-compatible signature; actual stateful stop logic is evaluated from run()
        # via attributes set on this function.
        state = getattr(ActiveLearning, "_stop_state", None)
        if not isinstance(state, dict):
            return False, "continue"

        labeled_count = int(state.get("labeled_count", 0) or 0)
        selected_mean_uncertainty = state.get("selected_mean_uncertainty", None)

        # --- Plateau stop on selected-batch uncertainty ---
        if selected_mean_uncertainty is None:
            return False, "continue"

        eps = ActiveLearning.hyper_params.get("uncertainty_plateau_eps", 0.0)
        patience = ActiveLearning.hyper_params.get("uncertainty_plateau_patience", 1)
        if not isinstance(eps, (int, float)):
            eps = 0.0
        if not isinstance(patience, int) or patience < 0:
            patience = 1

        # Skip uncertainty plateau check if disabled (patience = 0)
        if patience == 0:
            return False, "continue"

        prev = state.get("prev_selected_unc")
        streak = int(state.get("plateau_streak", 0) or 0)
        if prev is not None:
            try:
                drop = float(prev) - float(selected_mean_uncertainty)
            except Exception:
                drop = 0.0
            if drop < float(eps):
                streak += 1
            else:
                streak = 0

        state["prev_selected_unc"] = float(selected_mean_uncertainty)
        state["plateau_streak"] = streak

        if streak >= patience:
            return True, "uncertainty_plateau"

        return False, "continue"

    @staticmethod
    def random_sampling():
        N = ActiveLearning.hyper_params.get("N", 5)
        unlabelled = database.get_unlabelled_samples()
        if len(unlabelled) < N:
            return unlabelled
        return random.sample(unlabelled, N)

    @staticmethod
    def stratified_subsample(samples, n, label_index=3, seed=42):
        """Pick ~equal number of samples per class from an already-ranked candidate list.

        - Works as a *post-processing* step: upstream algorithm decides candidate ranking.
        - Preserves within-class order from `samples` (so scoring is respected inside each class).
        - If some classes have insufficient samples, fills remaining slots from the leftover pool.
        """
        if not samples:
            return samples
        if n is None or n <= 0:
            return []
        if len(samples) <= n:
            return samples

        rng = random.Random(seed)

        by_label = {}
        label_order = []
        for s in samples:
            label = None
            try:
                label = s[label_index]
            except Exception:
                label = None
            if label not in by_label:
                by_label[label] = []
                label_order.append(label)
            by_label[label].append(s)

        # Deterministic but shuffled distribution of remainder to avoid always favoring same labels
        labels = list(label_order)
        rng.shuffle(labels)

        k = len(labels)
        base = n // k
        rem = n % k

        quotas = {lab: base for lab in labels}
        for lab in labels[:rem]:
            quotas[lab] += 1

        picked = []
        leftovers = []
        for lab in label_order:
            rows = by_label.get(lab, [])
            q = quotas.get(lab, 0)
            picked.extend(rows[:q])
            leftovers.extend(rows[q:])

        if len(picked) < n:
            need = n - len(picked)
            picked.extend(leftovers[:need])

        return picked[:n]

    @staticmethod
    def set_seeds(seed: int = 42, deterministic: bool = False):
        if seed is not None:
            random.seed(seed)
        try:
            import numpy as np
            if seed is not None:
                np.random.seed(seed)
        except Exception:
            pass
        try:
            import torch
            if seed is not None:
                torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            if deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except Exception:
            pass

    @staticmethod
    def uncertainty_sampling():
        return database.uncertainty_sampling_selection(ActiveLearning.hyper_params["N"])

    @staticmethod
    def diversity_sampling():
        return database.diversity_sampling_selection(ActiveLearning.hyper_params["N"])

    @staticmethod
    def query_by_comitee(max_samples=MAX_SAMPLES):
        import math

        N = ActiveLearning.hyper_params.get("N", 5)
        models = ActiveLearning.model_list

        # Store full probability distributions per committee member per pool_id.
        # committee_probs[model_idx][pool_id] = probs (list[float])
        committee_probs = {}

        def _normalize_probs(p):
            try:
                probs = [float(x) for x in (p or [])]
                s = sum(probs)
                if s <= 0:
                    return None
                return [x / s for x in probs]
            except Exception:
                return None

        def _kl_div(p, q, eps=1e-12):
            # KL(p || q)
            kl = 0.0
            for pi, qi in zip(p, q):
                pi = float(pi)
                qi = float(qi)
                if pi <= 0:
                    continue
                kl += pi * math.log(pi / (qi + eps) + eps)
            return float(kl)

        def _js_div(probs_list):
            # Jensen-Shannon divergence across multiple distributions.
            # We use mean distribution M and average KL(P_i || M).
            if not probs_list or len(probs_list) < 2:
                return None
            k = len(probs_list[0])
            m = [0.0] * k
            for p in probs_list:
                for i in range(k):
                    m[i] += float(p[i])
            denom = float(len(probs_list))
            m = [x / denom for x in m]
            js = 0.0
            for p in probs_list:
                js += _kl_div(p, m)
            return float(js / denom)

        try:
            # Load each model & make predictions on (up to) max_samples unlabelled pool items.
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
                model_probs = {}

                while True:
                    batch = database.get_unlabelled_samples(batch_size, page * batch_size)
                    if not batch or (max_samples is not None and total_processed >= max_samples):
                        break

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
                            probs = _normalize_probs(res.get("probs"))
                            if probs is not None:
                                model_probs[pool_id] = probs
                        except Exception as e:
                            print(f"[committee] prediction error member {idx} id {pool_id}: {e}")

                    total_processed += len(batch)
                    page += 1

                if model_probs:
                    committee_probs[idx] = model_probs

            # Aggregate per pool_id and compute disagreement.
            probs_by_id = {}
            for _, model_map in committee_probs.items():
                for pid, probs in model_map.items():
                    probs_by_id.setdefault(pid, []).append(probs)

            scored = []
            for pid, probs_list in probs_by_id.items():
                if not probs_list or len(probs_list) < 2:
                    continue
                js = _js_div(probs_list)
                if js is None:
                    continue
                # Tie-breaker: encourage samples whose mean distribution is uncertain.
                # (entropy of mean probs)
                k = len(probs_list[0])
                m = [0.0] * k
                for p in probs_list:
                    for i in range(k):
                        m[i] += float(p[i])
                denom = float(len(probs_list))
                m = [x / denom for x in m]
                mean_entropy = -sum([pi * math.log(pi + 1e-12) for pi in m if pi > 0])
                # Normalize entropy to [0, 1] by dividing by log(K)
                ent_norm = None
                try:
                    ent_norm = float(mean_entropy) / float(math.log(float(k) + 1e-12))
                except Exception:
                    ent_norm = None
                scored.append((pid, (float(js), float(mean_entropy), ent_norm)))

            scored.sort(key=lambda x: (x[1][0], x[1][1]), reverse=True)
            selected_ids = [s[0] for s in scored[:N]]

            if not selected_ids:
                return []
            selected_samples = []
            for sample in database.pool:
                if sample[0] in selected_ids:
                    selected_samples.append(sample)

            return selected_samples
        finally:
            pass

    @staticmethod
    def run(function_algorithm, max_samples=None, test_samples=None, test_sample_limit=TEST_SAMPLE_LIMIT, test_from_db=True, base_train_size=BASE_TRAIN_SIZE):
        # Make runs comparable
        try:
            ActiveLearning.set_seeds(
                seed=ActiveLearning.hyper_params.get("seed", 42),
                deterministic=ActiveLearning.hyper_params.get("deterministic", False),
            )
        except Exception as e:
            print(f"Warning: failed to set seeds: {e}")

        # tum set
        all_labeled_samples = []

        # Ensure base classifier exists
        os.makedirs(ActiveLearning.RUNS_BASE, exist_ok=True)

        # If base_dir exists but is not a valid HF model folder, remove it so we can rebuild.
        try:
            if os.path.exists(ActiveLearning.BASE_DIR):
                cfg_path = os.path.join(ActiveLearning.BASE_DIR, "config.json")
                if not os.path.exists(cfg_path):
                    raise RuntimeError("missing config.json")
                try:
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                    if not isinstance(cfg, dict) or "model_type" not in cfg:
                        raise RuntimeError("config.json missing model_type")
                except Exception as e:
                    raise RuntimeError(f"invalid config.json: {e}")
        except Exception as e:
            try:
                print(f"Warning: invalid base classifier folder ({e}); removing {ActiveLearning.BASE_DIR} and rebuilding...")
                shutil.rmtree(ActiveLearning.BASE_DIR, ignore_errors=True)
            except Exception as e2:
                print(f"Warning: failed to remove invalid base classifier folder: {e2}")

        if not os.path.exists(ActiveLearning.BASE_DIR):
            print("Base classifier not found. Initializing and training...")

            ActiveLearning.initialize_and_train_base_classifier(
                train_size=base_train_size,          # 🔧 PARAMETRE
                base_dir=ActiveLearning.BASE_DIR,
                seed=ActiveLearning.hyper_params.get("seed"),
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

        try:
            base_model_id = getattr(JobClassifierTrainer, "model_name", None) or "unknown"
        except Exception:
            base_model_id = "unknown"

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

        # active_learning.py içinde (run metodunda)
        # ... BASE classifier evaluation sonrası ...
        # --- Remote DB insert (try remote first, then fallback to local RAM) ---
        try:
            avg_uncertainty = None
            if test_samples:
                avg_uncertainty = ActiveLearning.compute_avg_uncertainty_on_samples(base_trainer, test_samples)
            if avg_uncertainty is None:
                scores = database.get_all_uncertainty_scores()
                avg_uncertainty = sum([float(x) for x in scores]) / len(scores) if scores else None
            metrics = {"accuracy": base_accuracy, "avg_uncertainty": avg_uncertainty} if base_accuracy is not None else {"avg_uncertainty": avg_uncertainty}
            params = {
                "run_model_dir": ActiveLearning.BASE_DIR,
                "test_folder": test_folder,
                "method": "BASE",
                "DATA_SIZE": data_size,
            }

            # Önce remote DB'ye kaydetmeyi dene
            inserted_remote_id = database.insert_test_result_remote(
                test_id=test_id,
                iteration_no=0,
                model_name=ActiveLearning.normalize_model_name(base_model_id),
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
            
            if inserted_remote_id:
                print(f"Inserted BASE classifier result to remote DB with ID: {inserted_remote_id}")
            else:
                # Remote başarısız olursa, local RAM'e kaydet
                inserted_local_id = database.insert_test_result(
                    test_id=test_id,
                    iteration_no=0,
                    model_name=ActiveLearning.normalize_model_name(base_model_id),
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
                print(f"Inserted BASE classifier result to local RAM DB with ID: {inserted_local_id}")

        except Exception as e:
            print(f"Warning: failed to insert BASE result to DB: {e}")

        # Base accuracy artık referans noktası
        previous_accuracy = base_accuracy

        # Stop-state for plateau on selected uncertainty
        ActiveLearning._stop_state = {"prev_selected_unc": None, "plateau_streak": 0}

        while True:
            print(f"\n--- Iteration {iteration} ---")
            # Update stop-state before checks
            ActiveLearning._stop_state["labeled_count"] = len(all_labeled_samples)
            ActiveLearning._stop_state["selected_mean_uncertainty"] = None

            # Do not stop before evaluation; stop decisions are applied after training+evaluation
            stop_after_iteration = False
            stop_reason = "continue"
            
            # 1. Model tahmini (use current_model_dir)
            ActiveLearning.model_predict(max_samples, model_dir=current_model_dir)

            # 2. Algoritma ile etiketlenecek ornekleri sec
            selected_samples = function_algorithm()

            # QBC (ve genel) güvenlik: bazen seçim boş dönebilir (committee member fail vb.)
            # Bu durumda pipeline'ın boşa düşmemesi için fallback random seçim yap.
            if not selected_samples:
                fallback_n = ActiveLearning.hyper_params.get("N", 0) or 0
                if fallback_n > 0:
                    selected_samples = ActiveLearning.fallback_random_unlabelled(fallback_n)
            if ActiveLearning.hyper_params.get("stratified_batch", False):
                try:
                    selected_samples = ActiveLearning.stratified_subsample(
                        selected_samples,
                        ActiveLearning.hyper_params.get("N", len(selected_samples)),
                        label_index=3,
                        seed=42,
                    )
                except Exception as e:
                    print(f"Warning: stratified batch selection failed: {e}")
            print(f"Selected {len(selected_samples)} samples for labeling")

            # Selected-batch uncertainty (used for monitoring + plateau stop)
            mean_selected_unc = ActiveLearning.compute_mean_uncertainty_for_selected_samples(
                selected_samples,
                model_dir=current_model_dir,
            )
            if mean_selected_unc is not None:
                print(f"[signal] mean_uncertainty(selected_batch)={mean_selected_unc:.6f}")

            # Plateau check via central stop function
            ActiveLearning._stop_state["labeled_count"] = len(all_labeled_samples)
            ActiveLearning._stop_state["selected_mean_uncertainty"] = mean_selected_unc
            stop_plateau, plateau_reason = ActiveLearning.check_stop_condition(test_folder, iteration, None, None)
            if stop_plateau:
                stop_after_iteration = True
                stop_reason = plateau_reason

            # ara adim: veriyi etiketle
            labeled_samples = ActiveLearning.prep_labels(selected_samples)
            all_labeled_samples.extend(labeled_samples)
            print(f"Total labeled samples so far: {len(all_labeled_samples)}")

            # Budget may become true right after labeling
            ActiveLearning._stop_state["labeled_count"] = len(all_labeled_samples)
            ActiveLearning._stop_state["selected_mean_uncertainty"] = mean_selected_unc
            # Do not stop immediately after labeling; train+evaluate once with the final labeled set.

            # 3. Modeli eğit / güncelle
            # Train and save into the run-specific model folder (overwrite each iteration)
            try:
                trainer_obj, transformers_trainer = ActiveLearning.train_iterate(
                    all_labeled_samples,
                    source_model_dir=current_model_dir,
                    save_dir=run_model_dir,
                )
                print(f"Model trained successfully on {len(all_labeled_samples)} samples")
            except Exception as e:
                print(f"ERROR in train_iterate: {e}")
                # If training fails, break the loop
                break

            # After training, switch prediction to the latest model in run_model_dir
            current_model_dir = run_model_dir

            # 4. Accuracy kontrolü
            new_accuracy = None
            macro_f1 = None
            per_class_recalls = {}

            # If an external test set is provided, evaluate on it
            if test_samples:
                try:
                    print(f"Evaluating on test set ({len(test_samples)} samples)...")
                    new_accuracy = ActiveLearning.evaluate_on_test_set(trainer_obj, test_samples)
                    print(f"Test accuracy: {new_accuracy}")

                    # === Minority Recall + Macro-F1 ===
                    try:
                        # test dataset'i tekrar hazırla (predict için)
                        test_ds, _ = trainer_obj.prepare_datasets_from_tuples(
                            test_samples,
                            description_index=1,
                            label_index=3,
                            split=False,
                        )

                        hf_trainer = getattr(trainer_obj, "trainer", None)
                        if hf_trainer is None:
                            raise AttributeError("trainer_obj.trainer is missing")

                        predictions = hf_trainer.predict(test_ds)
                        logits = predictions.predictions
                        labels = predictions.label_ids

                        macro_f1, per_class_recalls = compute_minority_metrics(
                            logits,
                            labels,
                            trainer_obj.label_encoder
                        )

                        print(
                            f"[ITER {iteration}] macro_f1={macro_f1:.4f} "
                            f"recalls={per_class_recalls}"
                        )

                    except Exception as e:
                        print(f"Warning: minority metrics computation failed: {e}")
                        macro_f1 = None
                        per_class_recalls = {}

                except Exception as e:
                    print(f"Warning: Evaluation failed: {e}")
                    new_accuracy = None

            # 6. Sonuçları CSV'ye yaz
            result_file = ActiveLearning.get_next_result_file(test_folder)

            # Decide stop after we have evaluation metrics
            max_iters = int(ActiveLearning.hyper_params.get("max_iterations") or 0)
            max_iter_stop = (max_iters > 0 and iteration >= max_iters)
            label_budget = ActiveLearning.hyper_params.get("label_budget")
            budget_stop = isinstance(label_budget, int) and label_budget > 0 and len(all_labeled_samples) >= label_budget

            acc_thr = ActiveLearning.hyper_params.get("accuracy_threshold", None)
            accuracy_stop = False
            if acc_thr is not None and new_accuracy is not None:
                try:
                    accuracy_stop = float(new_accuracy) >= float(acc_thr)
                except Exception:
                    accuracy_stop = False

            stop_condition = bool(stop_after_iteration or accuracy_stop or max_iter_stop or budget_stop)
            if stop_condition:
                if stop_after_iteration:
                    final_reason = stop_reason
                elif accuracy_stop:
                    final_reason = f"accuracy_threshold ({acc_thr})"
                elif budget_stop:
                    final_reason = "label_budget"
                else:
                    final_reason = f"max_iterations_reached ({max_iters})"
            else:
                final_reason = "continue"
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
                    "stop_condition": stop_condition,
                    "stop_reason": final_reason,
                })

# active_learning.py içinde (iteration loop içinde)

            # 7. Save result - try remote first, then fallback to local RAM
            try:
                scores = database.get_all_uncertainty_scores()
                avg_uncertainty_pool = sum([float(x) for x in scores]) / len(scores) if scores else None
                metrics = {
                    "accuracy": new_accuracy,
                    # Backward-compatibility: avg_uncertainty remains pool-level.
                    "avg_uncertainty": avg_uncertainty_pool,
                    "avg_uncertainty_pool": avg_uncertainty_pool,
                    "selected_samples_avg_uncertainty": mean_selected_unc,
                    "macro_f1": macro_f1,
                }
                
                # Add per-class recalls if available
                if per_class_recalls:
                    for key, value in per_class_recalls.items():
                        metrics[f"recall_{str(key).lower()}"] = value
                
                params = {
                    "run_model_dir": run_model_dir,
                    "test_folder": test_folder,
                    "previous_accuracy": previous_accuracy,
                    "method": method_name,
                    "DATA_SIZE": data_size,
                    "accuracy_threshold": acc_thr,
                }

                if method_name == "query_by_comitee":
                    params["committee_models"] = list(getattr(ActiveLearning, "model_list", []) or [])
                
                # Önce remote DB'ye kaydetmeyi dene
                inserted_id = database.insert_test_result_remote(
                    test_id=test_id,
                    iteration_no=iteration,
                    model_name=ActiveLearning.normalize_model_name(getattr(trainer_obj, "model_name", None) or base_model_id),
                    train_data_size=len(all_labeled_samples),
                    method=method_name,
                    data_size=data_size,
                    N=ActiveLearning.hyper_params.get("N"),
                    T=ActiveLearning.hyper_params.get("T"),
                    I=None,
                    metrics=metrics,
                    params=params,
                    run_by=os.getenv("USER") or os.getenv("USERNAME"),
                    notes=final_reason,
                )
                
                if inserted_id:
                    print(f"Inserted result to REMOTE DB: test_id={test_id}, iteration={iteration}, id={inserted_id}")
                else:
                    # Remote başarısız olursa, local RAM'e kaydet
                    inserted_id = database.insert_test_result(
                        test_id=test_id,
                        iteration_no=iteration,
                        model_name=ActiveLearning.normalize_model_name(getattr(trainer_obj, "model_name", None) or base_model_id),
                        train_data_size=len(all_labeled_samples),
                        method=method_name,
                        data_size=data_size,
                        N=ActiveLearning.hyper_params.get("N"),
                        T=ActiveLearning.hyper_params.get("T"),
                        I=None,
                        metrics=metrics,
                        params=params,
                        run_by=os.getenv("USER") or os.getenv("USERNAME"),
                        notes=final_reason,
                    )
                    print(f"Inserted result to LOCAL RAM DB: test_id={test_id}, iteration={iteration}, id={inserted_id}")
                    
            except Exception as e:
                print(f"Warning: failed to insert result row to DB: {e}")

            print(f"Iteration {iteration} results saved to {result_file}")

            if stop_condition:
                print(f"Stopping: {final_reason}")
                break

            iteration += 1
            previous_accuracy = new_accuracy

            # max_iterations stop is handled above by persisting the final iteration row.
        
        # Son olarak veriyi CSV'ye kaydet
        try:
            database.save_to_csv()
            print("Data saved to CSV files")
        except Exception as e:
            print(f"Warning: failed to save data to CSV: {e}")

if __name__ == '__main__':
    # DB initialization already loaded the dataset (balanced if available).
    print(f"Pool size: {len(database.pool)}")
    print(f"Test set size: {len(database.test_data)}")
    print(f"Labelled samples: {len(database.get_labeled_samples())}")
    
    methods = [
        # ActiveLearning.uncertainty_sampling,
        # ActiveLearning.diversity_sampling,
        # ActiveLearning.query_by_comitee,
        ActiveLearning.random_sampling,
    ]

    for m in methods:
        print(f"\n=== Running method: {m.__name__} ===")
        try:
            ActiveLearning.run(m, max_samples=MAX_SAMPLES, test_from_db=True)
        except Exception as e:
            print(f"Error running {m.__name__}: {e}")