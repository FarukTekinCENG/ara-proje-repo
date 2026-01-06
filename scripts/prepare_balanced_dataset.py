import argparse
import csv
import os
import random
from typing import Dict, List, Optional


def _write_log(lines: List[str]) -> None:
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log.txt")
    with open(log_path, "a", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")


def _download_raw_dataset(raw_csv_path: str) -> None:
    from datasets import load_dataset

    os.makedirs(os.path.dirname(raw_csv_path) or ".", exist_ok=True)
    ds = load_dataset("datastax/linkedin_job_listings")
    ds["train"].to_csv(raw_csv_path, index=False)


def _read_and_group(
    raw_csv_path: str,
    label_col: str,
    text_col: str,
    allowed_labels: List[str],
) -> Dict[str, List[dict]]:
    allowed_set = set(allowed_labels)
    by_label: Dict[str, List[dict]] = {lab: [] for lab in allowed_labels}

    with open(raw_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in CSV: {raw_csv_path}")
        if label_col not in reader.fieldnames:
            raise KeyError(f"label_col '{label_col}' not found in CSV columns: {reader.fieldnames}")
        if text_col not in reader.fieldnames:
            raise KeyError(f"text_col '{text_col}' not found in CSV columns: {reader.fieldnames}")

        for row in reader:
            label = (row.get(label_col) or "").strip()
            text = (row.get(text_col) or "").strip()
            if not label or not text:
                continue
            if label not in allowed_set:
                continue
            by_label[label].append(row)

    return by_label


def build_balanced_dataset(
    raw_csv_path: str,
    balanced_csv_path: str,
    label_col: str = "formatted_work_type",
    text_col: str = "description",
    allowed_labels: Optional[List[str]] = None,
    target_per_class: int = 500,
    seed: int = 42,
    force_download: bool = False,
) -> str:
    if allowed_labels is None:
        allowed_labels = [
            "Contract",
            "Full-time",
            "Internship",
            "Other",
            "Part-time",
            "Temporary",
            "Volunteer",
        ]

    os.makedirs(os.path.dirname(raw_csv_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(balanced_csv_path) or ".", exist_ok=True)

    if force_download or not os.path.exists(raw_csv_path):
        print(f"[prepare_balanced_dataset] Downloading raw dataset to {raw_csv_path}...")
        _download_raw_dataset(raw_csv_path)

    print(f"[prepare_balanced_dataset] Reading raw CSV: {raw_csv_path}")
    by_label = _read_and_group(raw_csv_path, label_col, text_col, allowed_labels)

    counts = {k: len(v) for k, v in by_label.items()}
    print(f"[prepare_balanced_dataset] raw_class_counts: {counts}")

    missing = [k for k, v in by_label.items() if len(v) == 0]
    if missing:
        raise RuntimeError(f"Missing labels in raw dataset: {missing}")

    if target_per_class <= 0:
        raise ValueError("target_per_class must be > 0")

    rng = random.Random(seed)

    balanced_rows: List[dict] = []
    for lab in allowed_labels:
        rows = by_label[lab]
        if len(rows) >= target_per_class:
            rng.shuffle(rows)
            balanced_rows.extend(rows[:target_per_class])
        else:
            # Upsample with replacement to reach target_per_class
            balanced_rows.extend(rng.choices(rows, k=target_per_class))

    rng.shuffle(balanced_rows)

    with open(balanced_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=balanced_rows[0].keys())
        writer.writeheader()
        writer.writerows(balanced_rows)

    # Compute output distribution from the written file (source of truth)
    out_counts = {lab: 0 for lab in allowed_labels}
    with open(balanced_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lab = (row.get(label_col) or "").strip()
            if lab in out_counts:
                out_counts[lab] += 1

    lines = [
        "[prepare_balanced_dataset] DONE",
        f"balanced_csv_path={balanced_csv_path}",
        f"raw_csv_path={raw_csv_path}",
        f"target_per_class={target_per_class}",
        f"rows_total={len(balanced_rows)}",
        f"raw_class_counts={counts}",
        f"balanced_class_counts={out_counts}",
        "",
    ]

    for ln in lines:
        print(ln)

    _write_log(lines)

    return balanced_csv_path


def build_intelligent_balanced_dataset(
    raw_csv_path: str,
    balanced_csv_path: str,
    target_total_size: int = 15000,
    label_col: str = "formatted_work_type",
    text_col: str = "description",
    allowed_labels: Optional[List[str]] = None,
    seed: int = 42,
    force_download: bool = False,
) -> str:
    """
    Mod 2: Intelligent balanced dataset creation
    Strategy: Take all minority classes, then fill with medium classes, finally with majority class
    """
    if allowed_labels is None:
        allowed_labels = [
            "Contract",
            "Full-time",
            "Internship",
            "Other",
            "Part-time",
            "Temporary",
            "Volunteer",
        ]

    os.makedirs(os.path.dirname(raw_csv_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(balanced_csv_path) or ".", exist_ok=True)

    if force_download or not os.path.exists(raw_csv_path):
        print(f"[prepare_balanced_dataset] Downloading raw dataset to {raw_csv_path}...")
        _download_raw_dataset(raw_csv_path)

    print(f"[prepare_balanced_dataset] Reading raw CSV: {raw_csv_path}")
    by_label = _read_and_group(raw_csv_path, label_col, text_col, allowed_labels)

    counts = {k: len(v) for k, v in by_label.items()}
    print(f"[prepare_balanced_dataset] raw_class_counts: {counts}")

    missing = [k for k, v in by_label.items() if len(v) == 0]
    if missing:
        raise RuntimeError(f"Missing labels in raw dataset: {missing}")

    if target_total_size <= 0:
        raise ValueError("target_total_size must be > 0")

    # Sort labels by count (ascending) - minority to majority
    labels_by_count = sorted(allowed_labels, key=lambda x: len(by_label[x]))
    
    rng = random.Random(seed)
    balanced_rows: List[dict] = []
    remaining_size = target_total_size
    
    print(f"[prepare_balanced_dataset] Target total size: {target_total_size}")
    print(f"[prepare_balanced_dataset] Labels sorted by frequency: {labels_by_count}")
    
    # First pass: Take all minority classes
    minority_classes = []
    majority_classes = []
    
    for label in labels_by_count:
        available_rows = by_label[label]
        if len(available_rows) <= remaining_size and len(balanced_rows) + len(available_rows) < target_total_size:
            # This is a minority class - take all
            balanced_rows.extend(available_rows)
            remaining_size -= len(available_rows)
            minority_classes.append(label)
            print(f"[prepare_balanced_dataset] Took ALL {len(available_rows)} samples from '{label}'")
        else:
            # This is a majority class
            majority_classes.append(label)
    
    print(f"[prepare_balanced_dataset] Minority classes taken: {minority_classes}")
    print(f"[prepare_balanced_dataset] Majority classes remaining: {majority_classes}")
    print(f"[prepare_balanced_dataset] Remaining size to distribute: {remaining_size}")
    
    # Second pass: Distribute remaining size equally among majority classes
    if majority_classes and remaining_size > 0:
        per_class_target = remaining_size // len(majority_classes)
        remainder = remaining_size % len(majority_classes)
        
        print(f"[prepare_balanced_dataset] Distributing {per_class_target} samples per majority class (+1 for {remainder} classes)")
        
        for i, label in enumerate(majority_classes):
            available_rows = by_label[label]
            target_for_this_class = per_class_target + (1 if i < remainder else 0)
            
            if len(available_rows) >= target_for_this_class:
                rng.shuffle(available_rows)
                selected_rows = available_rows[:target_for_this_class]
                balanced_rows.extend(selected_rows)
                print(f"[prepare_balanced_dataset] Took {len(selected_rows)} samples from '{label}'")
            else:
                # Not enough samples in this class, take all
                balanced_rows.extend(available_rows)
                print(f"[prepare_balanced_dataset] Took ALL {len(available_rows)} samples from '{label}' (insufficient for target)")
    
    # Shuffle final dataset
    rng.shuffle(balanced_rows)

    # Write to CSV
    with open(balanced_csv_path, "w", newline="", encoding="utf-8") as f:
        if balanced_rows:
            writer = csv.DictWriter(f, fieldnames=balanced_rows[0].keys())
            writer.writeheader()
            writer.writerows(balanced_rows)
        else:
            raise RuntimeError("No balanced rows created")

    # Compute output distribution
    out_counts = {}
    with open(balanced_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lab = (row.get(label_col) or "").strip()
            out_counts[lab] = out_counts.get(lab, 0) + 1

    lines = [
        "[prepare_balanced_dataset] DONE - Mod 2: Intelligent Balanced",
        f"balanced_csv_path={balanced_csv_path}",
        f"raw_csv_path={raw_csv_path}",
        f"target_total_size={target_total_size}",
        f"rows_total={len(balanced_rows)}",
        f"raw_class_counts={counts}",
        f"balanced_class_counts={out_counts}",
        "",
    ]

    for ln in lines:
        print(ln)

    _write_log(lines)

    return balanced_csv_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=[1, 2], type=int, default=1, 
                       help="Mode 1: Equal per class, Mode 2: Intelligent balanced")
    parser.add_argument("--raw_csv_path", default="./data/job_postings.csv")
    parser.add_argument("--balanced_csv_path", default="./data/balanced_dataset.csv")
    parser.add_argument("--label_col", default="formatted_work_type")
    parser.add_argument("--text_col", default="description")
    parser.add_argument("--target_per_class", type=int, default=500)
    parser.add_argument("--target_total_size", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_download", action="store_true")
    parser.add_argument("--log_append", action="store_true")

    args = parser.parse_args()

    # If not appending, reset log file on each run
    if not args.log_append:
        log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log.txt")
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("")
        except Exception:
            pass

    if args.mode == 1:
        build_balanced_dataset(
            raw_csv_path=args.raw_csv_path,
            balanced_csv_path=args.balanced_csv_path,
            label_col=args.label_col,
            text_col=args.text_col,
            target_per_class=args.target_per_class,
            seed=args.seed,
            force_download=args.force_download,
        )
    elif args.mode == 2:
        build_intelligent_balanced_dataset(
            raw_csv_path=args.raw_csv_path,
            balanced_csv_path=args.balanced_csv_path,
            target_total_size=args.target_total_size,
            label_col=args.label_col,
            text_col=args.text_col,
            allowed_labels=None,  # Use default
            seed=args.seed,
            force_download=args.force_download,
        )


if __name__ == "__main__":
    main()
