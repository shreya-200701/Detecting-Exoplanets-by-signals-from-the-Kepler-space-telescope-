# Detecting-Exoplanets-by-signals-from-the-Kepler-space-telescope-
Use of Machine Learning to look at signals from the Kepler space telescope and decide whether a signal is likely to be a genuine exoplanet candidate or a false positive.
# Exoplanet Candidate Classification

A machine learning pipeline that classifies Kepler Objects of Interest (KOIs) as either **confirmed/candidate exoplanets** or **false positives**, using a Random Forest classifier trained on 54 astronomical measurement features.

## 📌 Overview

This project was built for a hackathon and predicts whether a detected transit signal corresponds to a real exoplanet (confirmed or candidate) or is a false positive, based on features derived from NASA's Kepler mission data (KOI - Kepler Objects of Interest).

- **Task:** Binary classification
- **Positive class (1):** `CONFIRMED` + `CANDIDATE`
- **Negative class (0):** `FALSE POSITIVE`
- **Model:** Random Forest Classifier (scikit-learn)

## 📂 Dataset

| File | Description |
|---|---|
| `traink.csv` | Training data — 7,651 rows × 54 features + `target` label |
| `testk.csv` | Holdout test data — 1,913 rows × 54 features (no labels, used for final prediction/submission) |
| `metadatak.csv` | Dataset metadata (row counts, split info, class definitions) |

**Note:** `testk.csv` does not contain ground-truth labels. Model evaluation is done via cross-validation on `traink.csv`; `testk.csv` is used only to generate final predictions for submission.

### Features
54 numeric features derived from Kepler light curve and stellar measurements, including:
- Orbital parameters (`koi_period`, `koi_duration`, `koi_depth`, `koi_impact`, etc.)
- Stellar properties (`koi_steff`, `koi_srad`, `koi_smass`, `koi_slogg`, etc.)
- Photometric magnitudes (`koi_kepmag`, `koi_gmag`, `koi_rmag`, etc.)
- Signal statistics (`koi_model_snr`, `koi_max_sngle_ev`, `koi_fwm_stat_sig`, etc.)

Some features contain missing values, handled via median imputation (fit on training data only, to avoid leakage).

## 🛠️ Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

Developed and tested in **Google Colab**.

## 🚀 Usage

1. Open the notebook in Google Colab.
2. Run all cells — you'll be prompted to upload `traink.csv`, `testk.csv`, and `metadatak.csv`.
3. The pipeline will:
   - Load and explore the data
   - Handle missing values via median imputation
   - Run 5-fold stratified cross-validation (accuracy, precision, recall, F1, ROC-AUC)
   - Train a final Random Forest model on the full training set
   - Plot feature importances
   - Generate predictions on the test set
   - Export `submission.csv` for download

## 📊 Model

**Random Forest Classifier**
```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42
)
```

Performance is evaluated using 5-fold stratified cross-validation on the training set, since the test set labels are withheld.

## 📁 Output

`submission.csv` — predictions for the test set:

| Column | Description |
|---|---|
| `row_id` | Row identifier matching `testk.csv` |
| `target` | Predicted class (0 = false positive, 1 = confirmed/candidate) |

## 📈 Possible Improvements

- Try gradient boosting models (XGBoost / LightGBM) for potentially higher accuracy
- Hyperparameter tuning via `GridSearchCV` or `RandomizedSearchCV`
- Feature selection based on importance rankings
- Handle class imbalance further with SMOTE or other resampling techniques

## 📄 License

Specify your license here (e.g. MIT).
