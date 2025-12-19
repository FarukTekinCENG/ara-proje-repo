#!/bin/bash
set -e

echo "PostgreSQL kuruluyor..."
apt-get update -qq
apt-get install -y postgresql postgresql-contrib

echo "Postgres servisi başlatılıyor..."
service postgresql start

echo "Database oluşturuluyor..."
sudo -u postgres psql <<EOF
CREATE DATABASE postgres;
EOF

echo "Schema uygulanıyor..."
sudo -u postgres psql postgres < schema.sql

echo "CSV import ediliyor..."
sudo -u postgres psql postgres <<EOF
\copy job_postings FROM '/content/postings.csv' WITH (
    FORMAT csv,
    HEADER true,
    DELIMITER ',',
    QUOTE '"'
);
EOF

echo "Bootstrap tamamlandı."

