WITH raw AS (
    SELECT
        dataset,
        pipeline,
        timestamp,
        ess_bulk_min,
        ess_tail_min,
        r_hat_max,
        divergences
    FROM read_csv_auto('data/diagnostics.csv')
)

SELECT
    pipeline,
    count(*) AS n_runs,

    sum(CASE WHEN ess_bulk_min < 400 THEN 1 ELSE 0 END) AS n_low_ess_bulk,
    round(min(ess_bulk_min), 1) AS worst_ess_bulk,

    sum(CASE WHEN ess_tail_min < 400 THEN 1 ELSE 0 END) AS n_low_ess_tail,
    round(min(ess_tail_min), 1) AS worst_ess_tail,

    sum(CASE WHEN r_hat_max > 1.01 THEN 1 ELSE 0 END) AS n_high_r_hat,
    round(max(r_hat_max), 4) AS worst_r_hat,

    sum(divergences) AS total_divergences
FROM raw
GROUP BY pipeline
ORDER BY pipeline
