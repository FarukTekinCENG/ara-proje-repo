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
    max_raw_rows: Optional[int] = 5000,
) -> str:
    os.makedirs(os.path.dirname(raw_csv_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(balanced_csv_path) or ".", exist_ok=True)

    if not os.path.exists(raw_csv_path):
        if not download_if_missing:
            raise FileNotFoundError(raw_csv_path)
        from datasets import load_dataset

        ds = load_dataset("datastax/linkedin_job_listings")
        ds["train"].to_csv(raw_csv_path, index=False)

        if max_raw_rows is not None:
            _truncate_csv_inplace(raw_csv_path, max_raw_rows=max_raw_rows)

    if os.path.exists(balanced_csv_path):
        return balanced_csv_path

    rng = random.Random(seed)

    with open(raw_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in CSV: {raw_csv_path}")

        if label_col not in reader.fieldnames:
            raise KeyError(f"label_col '{label_col}' not found in CSV columns: {reader.fieldnames}")
        if text_col not in reader.fieldnames:
            raise KeyError(f"text_col '{text_col}' not found in CSV columns: {reader.fieldnames}")

        by_label: Dict[str, List[dict]] = {}
        for row in reader:
            label = (row.get(label_col) or "").strip()
            text = (row.get(text_col) or "").strip()
            if not label or not text:
                continue
            by_label.setdefault(label, []).append(row)

    if not by_label:
        raise RuntimeError(f"No usable rows found in raw dataset: {raw_csv_path}")

    min_n = min(len(rows) for rows in by_label.values())
    if min_n <= 0:
        raise RuntimeError("Computed min class size is 0; cannot balance dataset")

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
