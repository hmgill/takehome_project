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
