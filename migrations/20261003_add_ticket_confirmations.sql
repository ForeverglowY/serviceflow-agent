BEGIN;

CREATE TABLE IF NOT EXISTS ticket_confirmations (
    ticket_id VARCHAR(50) PRIMARY KEY REFERENCES tickets(ticket_id),
    confirmed_by VARCHAR(100) NOT NULL,
    confirmed_at TIMESTAMPTZ NOT NULL,
    confirmation_message TEXT NOT NULL,
    application_snapshot JSONB NOT NULL
);

COMMIT;
