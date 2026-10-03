BEGIN;

ALTER TABLE logistics
    ADD COLUMN IF NOT EXISTS signed_at DATE;

-- The sample event records a confirmed signature for this order.
UPDATE logistics
SET signed_at = DATE '2026-07-25'
WHERE order_id = '20260721017'
  AND status = 'delivered'
  AND signed_at IS NULL;

COMMIT;
