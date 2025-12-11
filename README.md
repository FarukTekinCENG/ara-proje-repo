- .env file:
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=

- install dependencies:
pip install -r requirements.txt

- init postgres db with the tables under 'table_def/job_postings.sql'
