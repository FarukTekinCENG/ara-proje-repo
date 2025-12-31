#!/bin/bash
set -e

echo "PostgreSQL kuruluyor..."
apt-get update -qq
apt-get install -y postgresql postgresql-contrib

echo "Postgres servisi başlatılıyor..."
service postgresql start

echo "Schema uygulanıyor..."
sudo -u postgres psql postgres < job_postings.sql

echo "Train-Test Split..."
sudo -u python -m scripts.split_pool --fraction 0.2 --seed 42 --yes

echo "Bootstrap tamamlandı."
