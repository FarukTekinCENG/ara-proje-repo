#!/bin/bash
set -e

echo "Dataset HuggingFace'den indiriliyor ve CSV'ye yazılıyor..."

python << 'EOF'
from datasets import load_dataset

dataset = load_dataset("datastax/linkedin_job_listings")
dataset["train"].to_csv("/tmp/postings.csv", index=False)

print("CSV hazır:", "/tmp/postings.csv")
print("Satır sayısı:", len(dataset["train"]))
EOF

echo "PostgreSQL kuruluyor..."
apt-get update -qq
apt-get install -y postgresql postgresql-contrib

echo "Postgres servisi başlatılıyor..."
service postgresql start

echo "Schema uygulanıyor..."
sudo -u postgres psql postgres < table_def/job_postings.sql

echo "Train-Test Split..."
python -m scripts.split_pool --fraction 0.2 --seed 42 --yes

echo "Bootstrap tamamlandı."
