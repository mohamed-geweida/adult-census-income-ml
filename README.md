# Adult Census Income — ML Analysis

This repository contains an end-to-end exploratory and modeling notebook for the Adult Census Income dataset (predicting whether income > 50K).

## Contents

- `adult_census_income.ipynb` — main Jupyter notebook with data loading, cleaning, EDA, preprocessing, PCA, and a Random Forest model.
- `adult.csv` — dataset (not tracked here if large; place the CSV at the repo root to run the notebook).

## Project overview

The notebook walks through:

1. Loading and overview of the data
2. Handling missing values ("?" → NA, imputation for categorical fields)
3. Exploratory Data Analysis (education, age, hours/week, gender, correlation heatmap)
4. Feature engineering and preprocessing (scaling numeric features, one-hot encoding categoricals via ColumnTransformer)
5. Dimensionality reduction using PCA
6. Model building (Random Forest) and evaluation
7. Short clustering example (KMeans) for exploratory purposes

## Quick start

1. Clone the repo and place `adult.csv` at the repository root.
2. Create a Python environment and install dependencies.

PowerShell (Windows):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt  # or install packages listed below
```

If you don't have a `requirements.txt`, install minimal packages:

```powershell
pip install numpy pandas matplotlib seaborn scikit-learn jupyterlab
```

3. Start Jupyter Lab / Notebook and open `adult_census_income.ipynb`.

```powershell
jupyter lab
```

4. Run the notebook cells in order. The EDA and modeling cells are split so you can run sections independently.

## Notes on reproducibility

- The notebook uses fixed random seeds for PCA and RandomForest where applicable to make results reproducible.
- Preprocessing is implemented with a `ColumnTransformer` so it can be exported as part of a pipeline for production use.

## Results

The notebook prints model accuracy (Random Forest). For a submission-ready report, consider adding:

- Cross-validation and hyperparameter tuning (GridSearch or RandomizedSearch)
- Confusion matrix, precision/recall/F1, and ROC curve
- Saved model and preprocessing pipeline (joblib)

## Next steps / TODO

- Add `requirements.txt` with pinned versions.
- Add a short evaluation notebook or export visual report (HTML/PDF).
- Improve feature engineering (interactions, target encoding where appropriate).

## License

This repository is provided as-is for educational and demonstration purposes.

---

If you want, I can also create a pinned `requirements.txt`, run the notebook to capture final accuracy, or export the notebook to HTML for submission.
