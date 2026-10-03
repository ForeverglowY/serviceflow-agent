BEGIN;

ALTER TABLE products
    ADD COLUMN IF NOT EXISTS category VARCHAR(50);

-- The demo catalog explicitly classifies PRODUCT-001 as an in-ear earphone.
UPDATE products
SET category = 'in_ear_earphone'
WHERE product_id = 'PRODUCT-001'
  AND category IS NULL;

COMMIT;
