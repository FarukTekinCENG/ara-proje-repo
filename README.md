# .env file:
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=

#  init postgres db with the tables 
under 'table_def/job_postings.sql'

# split dataset train - test within db
python -m scripts.split_pool --fraction 0.2 --seed 42 --yes
