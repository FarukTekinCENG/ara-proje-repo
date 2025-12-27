import json
import csv

input_file = "/home/terminal/Downloads/04jobs/jobsnapshots-2025-08-27--2025-10-05.jsonl"
output_file = "job_descriptions.csv"

with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", newline='', encoding="utf-8") as f_out:
    writer = None
    for i, line in enumerate(f_in):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # bozuk satırları atla

        if writer is None:
            # CSV header'ı JSON keylerinden oluştur
            writer = csv.DictWriter(f_out, fieldnames=record.keys())
            writer.writeheader()

        writer.writerow(record)

        if (i + 1) % 10000 == 0:
            print(f"{i+1} satır işlendi...")

print("Tamamlandı!")

