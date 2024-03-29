BEGIN TRANSACTION;

CREATE TABLE result_raw (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code_version TEXT NOT NULL,
    date TEXT NOT NULL,
    hosts TEXT,
    threads INTEGER,
    params_sha TEXT NOT NULL,
    outliers_spec TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    algorithm_params TEXT NOT NULL,
    algorithm_version INT NOT NULL,
    dataset TEXT NOT NULL,
    dataset_params TEXT NOT NULL,
    dataset_version TEXT NOT NULL,
    shuffle_seed INTEGER NOT NULL,
    constraint_params TEXT NOT NULL,
    total_time_ms INTEGER NOT NULL,
    coreset_time_ms INTEGER NOT NULL,
    solution_time_ms INTEGER NOT NULL,
    distance_cnt INTEGER NOT NULL,
    oracle_cnt INTEGER NOT NULL,
    radius REAL NOT NULL,
    num_centers INTEGER NOT NULL,
    coreset_size INTEGER
);

CREATE VIEW recent_algorithm AS
SELECT
    algorithm,
    MAX(algorithm_version) AS algorithm_version
FROM
    result_raw
GROUP BY
    algorithm;

CREATE VIEW recent_dataset AS
SELECT
    dataset,
    MAX(dataset_version) AS dataset_version
FROM
    result_raw
GROUP BY
    dataset;

CREATE VIEW result AS
SELECT
    *
FROM
    result_raw NATURAL
    JOIN recent_dataset NATURAL
    JOIN recent_algorithm;

PRAGMA user_version = 1;

END TRANSACTION;