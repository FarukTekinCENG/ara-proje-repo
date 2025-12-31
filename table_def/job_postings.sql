CREATE TABLE job_postings (
    job_id                      TEXT,
    company_name                TEXT,
    title                       TEXT,
    description                 TEXT,
    max_salary                  TEXT,
    pay_period                  TEXT,
    location                    TEXT,
    company_id                  TEXT,
    views                       TEXT,
    med_salary                  TEXT,
    min_salary                  TEXT,
    formatted_work_type         TEXT,
    applies                     TEXT,
    original_listed_time        TEXT,
    remote_allowed              TEXT,
    job_posting_url             TEXT,
    application_url             TEXT,
    application_type            TEXT,
    expiry                      TEXT,
    closed_time                 TEXT,
    formatted_experience_level  TEXT,
    skills_desc                 TEXT,
    listed_time                 TEXT,
    posting_domain              TEXT,
    sponsored                   TEXT,
    work_type                   TEXT,
    currency                    TEXT,
    compensation_type           TEXT,
    normalized_salary           TEXT,
    zip_code                    TEXT,
    fips                        TEXT
);

\copy job_postings
FROM '/tmp/postings.csv'
WITH (
    FORMAT csv,
    HEADER true,
    DELIMITER ',',
    QUOTE '"'
);

ALTER TABLE job_postings
    ADD COLUMN id BIGSERIAL PRIMARY KEY;

CREATE TABLE pool (
    id                  BIGINT PRIMARY KEY,
    description         TEXT DEFAULT NULL,      
    is_labelled         TEXT DEFAULT 'FALSE',
    label               TEXT DEFAULT NULL,
    model_prediction    TEXT DEFAULT NULL,
    uncertainty_score   TEXT DEFAULT NULL,

    CONSTRAINT pool_job_fk
        FOREIGN KEY (id)
        REFERENCES job_postings(id)
        ON DELETE CASCADE
);

INSERT INTO pool (id, label, description)
SELECT id, formatted_work_type, description
FROM job_postings;
