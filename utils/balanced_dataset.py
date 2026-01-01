import csv
import os
import random
from typing import Dict, List, Optional


def ensure_balanced_dataset(
    raw_csv_path: str = "./data/job_postings.csv",
    balanced_csv_path: str = "./data/balanced_dataset.csv",
    label_col: str = "formatted_work_type",
    text_col: str = "description",
    seed: int = 42,
    download_if_missing: bool = True,
    max_raw_rows: Optional[int] = None,
    allowed_labels: Optional[List[str]] = None,
    min_expected_rows: int = 2000,
    min_raw_rows: int = 50000,
) -> str:
    os.makedirs(os.path.dirname(raw_csv_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(balanced_csv_path) or ".", exist_ok=True)

    def _download_full_raw() -> None:
        if not download_if_missing:
            raise FileNotFoundError(raw_csv_path)
        from datasets import load_dataset

        ds = load_dataset("datastax/linkedin_job_listings")
        ds["train"].to_csv(raw_csv_path, index=False)

        if max_raw_rows is not None:
            _truncate_csv_inplace(raw_csv_path, max_raw_rows=max_raw_rows)

    if not os.path.exists(raw_csv_path):
        _download_full_raw()
    else:
        # If an old run truncated the raw CSV (e.g. 5000 rows), rebuild it from the full dataset.
        try:
            approx_rows = _count_csv_rows_fast(raw_csv_path, max_rows_to_count=min_raw_rows)
            if approx_rows < min_raw_rows:
                _download_full_raw()
        except Exception:
            pass

    if os.path.exists(balanced_csv_path):
        try:
            if _is_balanced_csv_usable(
                balanced_csv_path,
                label_col=label_col,
                text_col=text_col,
                allowed_labels=allowed_labels,
                min_expected_rows=min_expected_rows,
            ):
                print(f"[balanced_dataset] Using existing balanced CSV: {balanced_csv_path}")
                return balanced_csv_path
        except Exception:
            pass

    print(f"[balanced_dataset] Building balanced CSV: {balanced_csv_path}")
    print(f"[balanced_dataset] Source raw CSV: {raw_csv_path}")

    rng = random.Random(seed)

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

    allowed_set = set(allowed_labels)

    with open(raw_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in CSV: {raw_csv_path}")

        if label_col not in reader.fieldnames:
            raise KeyError(f"label_col '{label_col}' not found in CSV columns: {reader.fieldnames}")
        if text_col not in reader.fieldnames:
            raise KeyError(f"text_col '{text_col}' not found in CSV columns: {reader.fieldnames}")

        by_label: Dict[str, List[dict]] = {lab: [] for lab in allowed_labels}
        for row in reader:
            label = (row.get(label_col) or "").strip()
            text = (row.get(text_col) or "").strip()
            if not label or not text:
                continue
            if label not in allowed_set:
                continue
            by_label[label].append(row)

    if not by_label or all(len(v) == 0 for v in by_label.values()):
        raise RuntimeError(f"No usable rows found in raw dataset: {raw_csv_path}")

    min_n = min(len(rows) for rows in by_label.values())
    counts_str = ", ".join([f"{k}={len(v)}" for k, v in by_label.items()])
    print(f"[balanced_dataset] class_counts: {counts_str}")
    print(f"[balanced_dataset] min_per_class={min_n} -> total={min_n * len(by_label)}")
    if min_n <= 0:
        missing = [lab for lab, rows in by_label.items() if not rows]
        raise RuntimeError(f"Computed min class size is 0; missing labels: {missing}")

    balanced_rows: List[dict] = []
    for rows in by_label.values():
        rng.shuffle(rows)
        balanced_rows.extend(rows[:min_n])

    rng.shuffle(balanced_rows)

    with open(balanced_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=balanced_rows[0].keys())
        writer.writeheader()
        writer.writerows(balanced_rows)

    return balanced_csv_path


def _truncate_csv_inplace(csv_path: str, max_raw_rows: int) -> None:
    tmp_path = csv_path + ".tmp"
    with open(csv_path, "r", encoding="utf-8") as fin, open(tmp_path, "w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in CSV: {csv_path}")
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for i, row in enumerate(reader):
            if i >= max_raw_rows:
                break
            writer.writerow(row)

    os.replace(tmp_path, csv_path)


def _count_csv_rows_fast(csv_path: str, max_rows_to_count: int) -> int:
    """Counts up to max_rows_to_count rows quickly (excluding header)."""
    n = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        # skip header
        _ = f.readline()
        for _ in f:
            n += 1
            if n >= max_rows_to_count:
                break
    return n


def _is_balanced_csv_usable(
    balanced_csv_path: str,
    label_col: str,
    text_col: str,
    allowed_labels: Optional[List[str]],
    min_expected_rows: int,
) -> bool:
    with open(balanced_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return False
        if label_col not in reader.fieldnames or text_col not in reader.fieldnames:
            return False

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

        allowed_set = set(allowed_labels)
        counts = {lab: 0 for lab in allowed_labels}
        total = 0
        for row in reader:
            label = (row.get(label_col) or "").strip()
            text = (row.get(text_col) or "").strip()
            if not text or label not in allowed_set:
                continue
            counts[label] += 1
            total += 1

    if total < min_expected_rows:
        return False
    if any(counts[lab] == 0 for lab in allowed_labels):
        return False
    return True
