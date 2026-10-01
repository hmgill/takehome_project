# Project configuration and outputs

The repository uses a single `config.toml` plus `src/project_config.py`
for shared paths and project settings. Python 3.11's standard-library
`tomllib` reads the configuration, so no additional config dependency is
required.

`src/runtime.py` centralizes CPU/thread and joblib temporary-directory
settings. Scripts call it before importing NumPy/scikit-learn/XGBoost.

## Output layout

```text
output/
├── logs/
│   ├── build_database.log
│   ├── analyze.log
│   ├── model.log
│   ├── model_svm.log
│   ├── model_xgboost.log
│   ├── create_predictions.log
│   ├── select_best_model.log
│   └── explain_shap.log
├── analysis/
│   ├── class_balance.csv
│   ├── image_characteristics.csv
│   ├── image_characteristics_by_split.csv
│   ├── qc_summary.csv
│   ├── cross_split_duplicates.csv
│   ├── duplicate_label_conflicts.csv
│   ├── unusual_images.csv
│   └── unusual_images.png
├── models/
│   ├── logistic/
│   ├── svm/
│   └── xgboost/
├── predictions/
├── model_selection/
├── shap/
│   ├── logistic/
│   └── xgboost/
├── mlflow/
├── splits/
│   ├── split_manifest.csv
│   └── split_report.json
└── tmp/
```

Part 1's generated dataset artifacts are written to `data/`:

```text
data/
├── pneumoniamnist.duckdb
├── pneumoniamnist_metadata.csv
└── image_catalog.sqlite3
```

The NPZ itself is user-supplied and lives outside the project; set
`PNEUMONIAMNIST_NPZ` to its path. Everything in `data/` and `output/` is
derived from the dataset and is excluded from git and the Docker image.

## Deduplication and splitting

All training and evaluation scripts (`model.py`, `model_svm.py`,
`model_xgboost.py`, `select_best_model.py`, `create_predictions.py`, and
the SHAP background in `explain_shap.py`) load data through
`src/data_splits.py`. With `[splits].deduplicate = true` (default):

1. the official train/val/test arrays are pooled;
2. exact duplicate images (SHA-256 of dtype, shape and pixels) are
   collapsed to their first occurrence;
3. duplicate groups with conflicting labels are dropped entirely
   (`label_conflict_policy = "drop"`) or keep their first copy
   (`"keep_first"`);
4. the unique images are re-split with a seeded stratified split whose
   proportions match the official splits by default.

No image can therefore appear twice in a split or in more than one split.
Training writes `output/splits/split_manifest.csv` (assigned split plus
source split/index and hash for every retained image) and
`split_report.json` (counts and a split fingerprint). Prediction CSVs
report the assigned `split` plus `source_split`/`source_index`, which
join to `main.image_metadata` in DuckDB.

The Streamlit catalog uses the same partition. NPZ images are cataloged
under their assigned `train`/`val`/`test` split; copies removed by
deduplication are cataloged as `excluded`, and uploaded or
folder-ingested images as `external`. Both can be selected in the app's
Split filter and are left out of the per-split performance metrics. After
upgrading, re-import the NPZ (or rerun `run_all.py --force`) so existing
catalog rows pick up the new categories.

Because the partition differs from the official MedMNIST splits, metrics
are not directly comparable to published benchmarks. Set
`deduplicate = false` to reproduce the official splits.

## Hyperparameter tuning

`model.py`, `model_svm.py` and `model_xgboost.py` share
`model_utils.tune_and_fit`, so every candidate is tuned identically:
the same seeded randomized search over `[modeling].search_iterations`
settings, scored by ROC AUC.

- `[modeling].cross_validation = false` (default): each setting is fit on
  train and scored on validation (`PredefinedSplit`). The winning setting
  is refit on **train only**, not train + validation.
- `cross_validation = true` or `--cv`: each setting is scored by
  stratified `cv_folds`-fold CV inside train; validation is not used for
  tuning.

Either way the validation split chooses the decision threshold (Youden's
J) and `select_best_model.py` compares candidates by validation ROC AUC;
the test split is scored once. In holdout mode validation also chose the
hyperparameters, so validation scores are slightly optimistic; the test
score stays unbiased.

## Model used by the app

`[app].model` (`logistic`, `svm`, `xgboost` or `best`), overridden by
`IMAGE_EXPLORER_MODEL`, selects the classifier for `image_pipeline.py`
and the Streamlit app. `best` reads
`output/model_selection/selected_model.json`. Each prediction stores the
`model_version` (SHA-256 of the model name, model file and metrics file),
so `image_pipeline.py --rescore` and the app's Re-score button can update
only predictions made by a different model.

## Augmentation

Augmentation is disabled by default. `src/augmentation_utils.py` contains
a conservative optional policy using small affine, brightness, and
contrast perturbations. It should be applied to the training split only.

For these 28x28 flattened-pixel models, augmentation is an experiment,
not part of the required baseline. Small translations/rotations can help
test robustness, but aggressive geometric transforms can damage medically
meaningful structure and can also hurt non-spatial models such as logistic
regression, SVM, and XGBoost.

## Streamlit application

Run `python -m streamlit run app.py` from the project root. The shared
`src/image_pipeline.py` service populates `data/image_catalog.sqlite3`.
This application catalog stores upload/source records, validated metadata,
thumbnail and inference feature BLOBs, and versioned predictions. It is
separate from the original research DuckDB tables. See `README.md` for the
end-to-end commands, schema semantics, limits and incremental behavior.
