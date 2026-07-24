WITH raw AS (
    SELECT
        dataset,
        pipeline,
        subject,
        session,
        brier,
        reliability,
        resolution,
        auroc,
        entropy
    FROM read_csv_auto('data/metrics.csv')
),

fold AS (
    SELECT
        dataset,
        pipeline,
        subject,
        avg(brier) AS f_brier,
        avg(reliability) AS f_reliability,
        avg(resolution) AS f_resolution,
        avg(auroc) AS f_auroc,
        avg(entropy) AS f_entropy
    FROM raw
    GROUP BY dataset, pipeline, subject
),

cv AS (
    SELECT
        dataset,
        pipeline,
        avg(f_brier) AS cv_brier,
        avg(f_reliability) AS cv_reliability,
        avg(f_resolution) AS cv_resolution,
        avg(f_auroc) AS cv_auroc,
        avg(f_entropy) AS cv_entropy
    FROM fold
    GROUP BY dataset, pipeline
),

pair AS (
    SELECT
        'CSPLDA' AS freq,
        'CSPBLDA' AS bayes
    UNION ALL
    SELECT
        'CSPSVM' AS freq,
        'CSPGP' AS bayes
    UNION ALL
    SELECT
        'TSLR' AS freq,
        'TSBLR' AS bayes
    UNION ALL
    SELECT
        'TSSVM' AS freq,
        'TSGP' AS bayes
    UNION ALL
    SELECT
        'SCNN' AS freq,
        'BSCNN' AS bayes
    UNION ALL
    SELECT
        'DCNN' AS freq,
        'BDCNN' AS bayes
),

diff AS (
    SELECT
        f.dataset,
        p.freq,
        p.bayes,
        b.cv_brier - f.cv_brier AS diff_brier,
        b.cv_reliability - f.cv_reliability AS d_reliability,
        b.cv_resolution - f.cv_resolution AS d_resolution,
        b.cv_auroc - f.cv_auroc AS diff_auroc,
        b.cv_entropy - f.cv_entropy AS diff_entropy
    FROM pair AS p
    INNER JOIN cv AS f ON p.freq = f.pipeline
    INNER JOIN cv AS b ON p.bayes = b.pipeline AND f.dataset = b.dataset
),

stat AS (
    SELECT
        dataset,

        avg(diff_brier) AS yi_brier,
        var_samp(diff_brier) / count(*) AS vi_brier,

        avg(d_reliability) AS yi_reliability,
        var_samp(d_reliability) / count(*) AS vi_reliability,

        avg(d_resolution) AS yi_resolution,
        var_samp(d_resolution) / count(*) AS vi_resolution,

        avg(diff_auroc) AS yi_auroc,
        var_samp(diff_auroc) / count(*) AS vi_auroc,

        avg(diff_entropy) AS yi_entropy,
        var_samp(diff_entropy) / count(*) AS vi_entropy
    FROM diff
    GROUP BY dataset
)

SELECT * FROM stat
ORDER BY dataset
