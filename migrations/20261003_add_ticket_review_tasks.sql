BEGIN;

CREATE TABLE IF NOT EXISTS ticket_review_tasks (
    ticket_id VARCHAR(50) PRIMARY KEY REFERENCES tickets(ticket_id),
    decision VARCHAR(30) NOT NULL,
    reason TEXT NOT NULL,
    status VARCHAR(30) NOT NULL DEFAULT 'pending',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

COMMIT;
