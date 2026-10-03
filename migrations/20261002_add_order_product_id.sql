BEGIN;

ALTER TABLE orders
    ADD COLUMN IF NOT EXISTS product_id VARCHAR(50);

-- Only unlinked legacy rows need a one-time name match. Abort if it is ambiguous.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM orders AS o
        LEFT JOIN products AS p ON p.name = o.product_name
        WHERE o.product_id IS NULL
        GROUP BY o.order_id
        HAVING COUNT(p.product_id) <> 1
    ) THEN
        RAISE EXCEPTION 'Some orders do not match exactly one product';
    END IF;
END
$$;

UPDATE orders AS o
SET product_id = p.product_id
FROM products AS p
WHERE o.product_id IS NULL
  AND o.product_name = p.name;

ALTER TABLE orders
    ALTER COLUMN product_id SET NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conrelid = 'orders'::regclass
          AND conname = 'orders_product_id_fkey'
    ) THEN
        ALTER TABLE orders
            ADD CONSTRAINT orders_product_id_fkey
            FOREIGN KEY (product_id) REFERENCES products(product_id);
    END IF;
END
$$;

COMMIT;
