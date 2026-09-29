CREATE SCHEMA IF NOT EXISTS analysis;

CREATE OR REPLACE VIEW analysis.class_balance AS
SELECT
    split,
    COUNT(*) AS total_images,
    COUNT(*) FILTER (WHERE label = 0) AS normal_images,
    COUNT(*) FILTER (WHERE label = 1) AS pneumonia_images,
    ROUND(
        100.0
        * COUNT(*) FILTER (WHERE label = 1)
        / COUNT(*),
        2
    ) AS pct_pneumonia
FROM main.image_metadata
GROUP BY split;


CREATE OR REPLACE VIEW analysis.image_characteristics AS
SELECT
    label,
    CASE
        WHEN label = 0 THEN 'normal'
        WHEN label = 1 THEN 'pneumonia'
    END AS class_name,
    COUNT(*) AS total_images,
    ROUND(AVG(mean_intensity), 2) AS avg_mean_intensity,
    ROUND(AVG(std_intensity), 2) AS avg_image_std_intensity,
    ROUND(MIN(mean_intensity), 2) AS min_mean_intensity,
    ROUND(MAX(mean_intensity), 2) AS max_mean_intensity,
    ROUND(MIN(std_intensity), 2) AS min_image_std_intensity,
    ROUND(MAX(std_intensity), 2) AS max_image_std_intensity
FROM main.image_metadata
GROUP BY label;


CREATE OR REPLACE VIEW analysis.image_characteristics_by_split AS
SELECT
    split,
    label,
    CASE
        WHEN label = 0 THEN 'normal'
        WHEN label = 1 THEN 'pneumonia'
    END AS class_name,
    COUNT(*) AS total_images,
    ROUND(AVG(mean_intensity), 2) AS avg_mean_intensity,
    ROUND(AVG(std_intensity), 2) AS avg_image_std_intensity
FROM main.image_metadata
GROUP BY
    split,
    label;


CREATE OR REPLACE VIEW analysis.qc_summary AS
SELECT
    COUNT(*) AS total_images,
    COUNT(*) FILTER (
        WHERE low_variance_flag
    ) AS low_variance_images,
    COUNT(*) FILTER (
        WHERE exact_duplicate_flag
    ) AS duplicate_records,
    COUNT(DISTINCT image_hash) FILTER (
        WHERE exact_duplicate_flag
    ) AS duplicate_groups,
    COUNT(*) FILTER (
        WHERE cross_split_duplicate_flag
    ) AS cross_split_duplicate_records,
    COUNT(DISTINCT image_hash) FILTER (
        WHERE cross_split_duplicate_flag
    ) AS cross_split_duplicate_groups
FROM main.image_metadata;


CREATE OR REPLACE VIEW analysis.cross_split_duplicates AS
SELECT
    image_hash,
    image_id,
    split,
    split_index,
    label,
    duplicate_count,
    duplicate_image_ids
FROM main.image_metadata
WHERE cross_split_duplicate_flag = TRUE;


CREATE OR REPLACE VIEW analysis.duplicate_label_conflicts AS
SELECT
    image_hash,
    COUNT(*) AS record_count,
    COUNT(DISTINCT label) AS distinct_label_count,
    STRING_AGG(
        CAST(label AS VARCHAR),
        ', ' ORDER BY label
    ) AS labels,
    STRING_AGG(
        split,
        ', ' ORDER BY split
    ) AS splits
FROM main.image_metadata
WHERE exact_duplicate_flag = TRUE
GROUP BY image_hash
HAVING COUNT(DISTINCT label) > 1;


CREATE OR REPLACE VIEW analysis.unusual_images AS
WITH ranked_images AS (
    SELECT
        image_id,
        split,
        split_index,
        label,
        mean_intensity,
        std_intensity,
        min_intensity,
        max_intensity,
        low_variance_flag,
        exact_duplicate_flag,
        ROW_NUMBER() OVER (
            ORDER BY mean_intensity ASC, image_id
        ) AS darkest_rank,
        ROW_NUMBER() OVER (
            ORDER BY mean_intensity DESC, image_id
        ) AS brightest_rank,
        ROW_NUMBER() OVER (
            ORDER BY std_intensity ASC, image_id
        ) AS lowest_contrast_rank
    FROM main.image_metadata
)

SELECT
    image_id,
    split,
    split_index,
    label,
    mean_intensity,
    std_intensity,
    min_intensity,
    max_intensity,
    low_variance_flag,
    exact_duplicate_flag,
    'darkest' AS criterion,
    darkest_rank AS selection_rank,
    'mean_intensity' AS metric_name,
    mean_intensity AS metric_value
FROM ranked_images
WHERE darkest_rank <= 3

UNION ALL

SELECT
    image_id,
    split,
    split_index,
    label,
    mean_intensity,
    std_intensity,
    min_intensity,
    max_intensity,
    low_variance_flag,
    exact_duplicate_flag,
    'brightest' AS criterion,
    brightest_rank AS selection_rank,
    'mean_intensity' AS metric_name,
    mean_intensity AS metric_value
FROM ranked_images
WHERE brightest_rank <= 3

UNION ALL

SELECT
    image_id,
    split,
    split_index,
    label,
    mean_intensity,
    std_intensity,
    min_intensity,
    max_intensity,
    low_variance_flag,
    exact_duplicate_flag,
    'lowest_contrast' AS criterion,
    lowest_contrast_rank AS selection_rank,
    'std_intensity' AS metric_name,
    std_intensity AS metric_value
FROM ranked_images
WHERE lowest_contrast_rank <= 3;
