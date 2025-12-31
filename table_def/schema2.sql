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
