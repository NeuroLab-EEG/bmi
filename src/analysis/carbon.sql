WITH raw AS (
    SELECT
        dataset,
        pipeline,
        subject,
        session,
        energy_consumed
    FROM read_csv_auto('data/carbon.csv')
),

fold AS (
    SELECT
        dataset,
        pipeline,
        subject,
        avg(energy_consumed) AS f_energy_consumed
    FROM raw
    GROUP BY dataset, pipeline, subject
),

per_dataset AS (
    SELECT
        dataset,
        pipeline,
        avg(f_energy_consumed) AS d_energy_consumed
    FROM fold
    GROUP BY dataset, pipeline
)

SELECT
    pipeline,
    count(*) AS n_datasets,
    round(percentile_cont(0.25) WITHIN GROUP (ORDER BY d_energy_consumed) * 1000, 3)
        AS q1_energy_consumed,
    round(percentile_cont(0.5) WITHIN GROUP (ORDER BY d_energy_consumed) * 1000, 3)
        AS q2_energy_consumed,
    round(percentile_cont(0.75) WITHIN GROUP (ORDER BY d_energy_consumed) * 1000, 3)
        AS q3_energy_consumed
FROM per_dataset
GROUP BY pipeline
ORDER BY pipeline
