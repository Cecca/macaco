BEGIN TRANSACTION;

ALTER TABLE
	result_raw
ADD
	COLUMN coreset_radius REAL;

PRAGMA user_version = 2;

END TRANSACTION;