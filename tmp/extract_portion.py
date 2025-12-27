import json
import pandas as pd

input_file = "/home/terminal/Downloads/04jobs/jobsnapshots-2025-08-27--2025-10-05.jsonl"
output_file = "job_descriptions_5000.jsonl"
n = 5000  # kaç kayıt alınacak

data = []
with open(input_file, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= n:
            break
        data.append(json.loads(line))

# DataFrame olarak kaydetmek isterseniz:
df = pd.DataFrame(data)
df.to_json(output_file, orient="records", lines=True)

print(f"İlk {n} kayıt {output_file} dosyasına kaydedildi.")

