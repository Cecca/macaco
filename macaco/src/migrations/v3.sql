BEGIN TRANSACTION;

ALTER TABLE
	result_raw
ADD
	COLUMN memory_coreset_kb INT64;

PRAGMA user_version = 3;

END TRANSACTION;