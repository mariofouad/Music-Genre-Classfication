# Music Genre Classification

CMPS344 Applied Data Science project for supervised music genre classification.

The project follows the submitted proposal: Spotify tabular audio features are the main training dataset, while FMA and GTZAN are supported as external real-world benchmark sources after genre-label harmonization. The old `rannem-backend` song embedder is intentionally not copied into this project; it is an audio embedding/recommendation service, not a finished Spotify genre classifier.

## Team

- Mario Emad Saleh Fouad - 1210158
- Mohamed ElSayed Shaaban - 1210025
- Moaz Mohamed ElSayed - 1210162
- Seif Wael - 1210243

## Data Files

Place raw datasets in `data/raw` using the paths configured in `configs/project.yaml`:

- `spotify_tracks.csv`: required primary dataset.
- `fma_tracks.csv`: optional FMA metadata file.
- `fma_echonest.csv`: optional FMA Echo Nest/audio feature file.
- `gtzan_features.csv`: optional GTZAN derived feature table.

The pipeline can run with Spotify only. FMA and GTZAN rows are added when their files exist and contain compatible genre/audio-feature columns.

## Setup

```powershell
poetry install
```

If Poetry is not installed:

```powershell
python -m pip install poetry
poetry install
```

## Common Commands

```powershell
make data       # ingest, clean, integrate, and validate raw datasets
make train      # engineer features, split data, train models, log MLflow runs
make evaluate   # evaluate the best model on test/external splits
make dashboard  # open the Streamlit dashboard
make test       # run pytest with coverage
make lint       # run ruff
```

Equivalent direct commands are available through Poetry scripts:

```powershell
poetry run music-genre-data --config configs/project.yaml
poetry run music-genre-train --config configs/project.yaml
poetry run music-genre-evaluate --config configs/project.yaml
```

Run the complete data, training, evaluation, and dashboard flow with:

```powershell
make serve
```

## CI/CD

GitHub Actions is configured in `.github/workflows/ci.yml`.

On pull requests and pushes to `main`, the CI job:

- validates `pyproject.toml` and `poetry.lock`
- installs dependencies with Poetry
- runs Ruff linting
- runs pytest with coverage
- smoke-tests the CLI entry points
- builds the Python package
- uploads `coverage.xml` and `dist/` as CI artifacts

On every **push to `main`**, after CI passes, the **Deploy gate** job installs from **`requirements.txt`** (the same install path Streamlit Community Cloud uses), compiles **`app.py`**, and imports the dashboard module so broken deploys are caught before Cloud rebuilds.

On pushes to `main` and manual workflow runs, the delivery job also uploads a submission artifact named `music-genre-classification-submission`. The artifact contains the code, configs, reports written in Markdown, tests, README, and project metadata. Large local outputs such as raw datasets, generated model files, MLflow runs, and generated JSON reports are excluded because they can be recreated with `make serve`.

### Continuous deployment (Streamlit)

[Streamlit Community Cloud](https://streamlit.io/cloud) redeploys the app when you push to the branch the app is pinned to (often `main`). Connect the GitHub repo once, set **Main file** to **`app.py`**, Python **3.11**, and keep **`requirements.txt`** committed (refresh it after dependency changes with `poetry export -f requirements.txt --output requirements.txt --without-hashes --only main`). You do not need a separate deploy token for basic GitHub → Cloud CD.

Before pushing, run the local equivalent of the main CI checks:

```powershell
make ci
```

To test the workflow on GitHub:

1. Commit and push the branch.
2. Open the repository on GitHub.
3. Go to **Actions**.
4. Open the **CI/CD** workflow run.
5. Confirm that **Lint, test, and build** passes.
6. If the run is on `main` or manually triggered, confirm that **Build submission artifact** passes.
7. Download artifacts from the workflow summary to inspect `ci-artifacts` and `music-genre-classification-submission`.

## Deploy the dashboard (Streamlit Community Cloud)

1. Push the repo to GitHub.
2. [Streamlit Community Cloud](https://streamlit.io/cloud) → **New app** → select repo/branch.
3. **Main file:** **`app.py`** (repo root — it fixes `sys.path` before importing `music_genre`; avoids Cloud import errors). **Python:** 3.11.
4. Add a root **`requirements.txt`**: install the export plugin if needed (`poetry self add poetry-plugin-export`), then  
   `poetry export -f requirements.txt --output requirements.txt --without-hashes --only main`  
   Commit and push `requirements.txt`.
5. Include **`models/best_model.joblib`** and **`data/processed/`** in the deployed tree (commit, Git LFS, or download at startup) so charts and the test-set demo work.

Use **`app.py`** as the Streamlit entry point. Alternatively append **`-e .`** to `requirements.txt` so `pip` installs the project as a package (then `src/music_genre/dashboard.py` works too).

## Outputs

- `data/interim/integrated_tracks.csv`: cleaned, harmonized data.
- `data/processed/model_ready.csv`: data with engineered features.
- `data/processed/splits`: train, validation, test, and external splits.
- `reports/validation_report.json`: row counts, missing values, class balance, source distribution, and acceptance checks.
- `reports/training_summary.json`: all model runs and selected best model.
- `reports/evaluation_metrics.json`: test and external benchmark metrics.
- `models/best_model.joblib`: best validation model.
- `mlruns`: MLflow experiment tracking artifacts.

## Implemented Models and Metrics

The training pipeline logs at least five classifiers:

- Dummy baseline
- Logistic Regression
- Random Forest
- Gradient Boosting
- Linear SVM

Each run logs accuracy, macro F1, weighted F1, top-3 accuracy when class probabilities are available, high-confidence accuracy, and costly misclassification rate.


