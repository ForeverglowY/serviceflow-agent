BEGIN;

ALTER TABLE tickets
    ADD COLUMN IF NOT EXISTS submission_id VARCHAR (32);

DO
$$
BEGIN
    IF
NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'tickets'::regclass
          AND conname = 'tickets_submission_id_key'
    ) THEN
ALTER TABLE tickets
    ADD CONSTRAINT tickets_submission_id_key UNIQUE (submission_id);
END IF;
END $$;

COMMIT;
