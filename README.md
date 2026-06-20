# Credit Risk Prediction

A machine learning project that predicts whether a loan applicant is a **good** or **bad** credit risk using the classic German Credit dataset. The pipeline covers exploratory data analysis, data cleaning, categorical encoding, and training/comparison of four classification algorithms, with the best model serialized for reuse.

## Overview

Financial institutions need to assess loan applicants quickly and consistently. This project builds a supervised classification pipeline that takes applicant attributes (age, job, housing, account balances, credit amount, loan duration, and purpose) and predicts credit risk (`good` / `bad`).

## Dataset

- **Source:** German Credit Data (1,000 records, 11 columns)
- **Target variable:** `Risk` (`good` / `bad`)
- **Features used:**
  - `Age`, `Job` (numeric)
  - `Sex`, `Housing`, `Saving accounts`, `Checking account`, `Purpose` (categorical)
  - `Credit amount`, `Duration` (numeric)
- Rows with missing values (mostly in `Saving accounts` and `Checking account`) are dropped, leaving **522 records** for modeling.

## Project Structure

```
.
├── File.ipynb                          # Main notebook: EDA, preprocessing, modeling
├── german_credit_data.csv              # Source dataset
├── Sex_label_encoder.pkl               # Saved LabelEncoder for "Sex"
├── Housing_label_encoder.pkl           # Saved LabelEncoder for "Housing"
├── Saving accounts_label_encoder.pkl   # Saved LabelEncoder for "Saving accounts"
├── Checking account_label_encoder.pkl  # Saved LabelEncoder for "Checking account"
├── Purpose_label_encoder.pkl           # Saved LabelEncoder for "Purpose"
├── target_label_encoder.pkl            # Saved LabelEncoder for "Risk" (target)
└── best_model.pkl                      # Best-performing trained model (generated on run)
```

## Methodology

1. **Exploratory Data Analysis**
   - Distribution and outlier checks for `Age`, `Credit amount`, `Duration`
   - Correlation analysis between numeric features
   - Categorical feature distributions, broken down by `Risk`
   - Cross-tabs of `Credit amount` by `Job`, `Sex`, `Housing`, and `Purpose`

2. **Preprocessing**
   - Dropped rows with missing values
   - Label-encoded all categorical features, saving each encoder individually (so the exact same transformation can be reapplied to new data at inference time)
   - Label-encoded the target (`Risk`) separately
   - 80/20 stratified train/test split

3. **Modeling**
   Four classifiers were trained and tuned via `GridSearchCV` (5-fold cross-validation, accuracy scoring):
   - Decision Tree
   - Random Forest
   - Extra Trees
   - XGBoost

   Each model's best estimator is evaluated on the held-out test set, and accuracy/classification reports are compared to select the best-performing model.

## Results

| Model          | Test Accuracy |
|----------------|---------------|
| Decision Tree  | 0.60          |
| Random Forest  | 0.67          |
| Extra Trees    | 0.61          |
| XGBoost        | *(run locally — see note below)* |

> Random Forest was the top performer among the models tested. Results will vary slightly depending on environment and library versions.

## Tech Stack

- Python
- pandas, NumPy
- scikit-learn
- XGBoost
- matplotlib, seaborn
- joblib (model/encoder serialization)

## Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib
```

### Running the project

1. Clone the repo and make sure `german_credit_data.csv` is in the same directory as the notebook.
2. Open `File.ipynb` in Jupyter and run all cells top to bottom.
3. The notebook will:
   - Perform EDA and display plots
   - Clean the data and encode categorical features (saving each encoder as a `.pkl`)
   - Train and compare all four models
   - Save the best-performing model as `best_model.pkl`

### Using the saved model

```python
import joblib
import pandas as pd

model = joblib.load("best_model.pkl")
le_target = joblib.load("target_label_encoder.pkl")

# Re-apply the same encoders used during training to any new categorical input
sex_encoder = joblib.load("Sex_label_encoder.pkl")
housing_encoder = joblib.load("Housing_label_encoder.pkl")
# ...etc for "Saving accounts", "Checking account", "Purpose"

prediction = model.predict(X_new)
print(le_target.inverse_transform(prediction))
```

## Note on XGBoost

The XGBoost cell is included in the notebook but requires the `xgboost` package installed locally to run. All other models (Decision Tree, Random Forest, Extra Trees) are fully tested and verified end-to-end.


## License

This project is for academic and educational purposes.
