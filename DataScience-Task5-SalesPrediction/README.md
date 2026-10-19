# Data Science Task 5 — Sales Prediction Using Python

**Internship track:** Oasis Infobyte — Data Science  
**Objective:** Predict product sales from advertising spend in TV, Radio, and Newspaper channels.

## Project contents

- `Sales_Prediction_Using_Python_Task5.ipynb` — main, explained Jupyter Notebook.
- `sales_prediction.py` — script version of the analysis.
- `requirements.txt` — Python dependencies.

## Dataset

Use the classic `Advertising.csv` dataset with four columns: `TV`, `Radio`, `Newspaper`, and `Sales`.

Dataset source: [Advertising.csv in the ISLR dataset repository](https://github.com/nguyen-toan/ISLR/blob/master/dataset/Advertising.csv).

The notebook/script looks for a local `Advertising.csv` first and otherwise tries the public raw GitHub URL. For the most reliable run, download the CSV (use GitHub's Raw/Download option) and put it in this folder beside the notebook/script.

## How to run

1. Install Python 3.10+ and open a terminal in this project folder.
2. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Launch Jupyter:

   ```bash
   jupyter notebook
   ```

4. Open `Sales_Prediction_Using_Python_Task5.ipynb` and run **Kernel → Restart Kernel and Run All Cells**.
5. Read the printed metrics and chart outputs. Complete the Final Findings section in the notebook using your actual outputs, then save it.

To run the script instead:

```bash
python sales_prediction.py
```

## Methodology

1. Data loading, required-column validation, missing-value and duplicate checks.
2. Exploratory data analysis: descriptive statistics, pairplot, three channel scatter plots, and correlation heatmap.
3. 80/20 train/test split with a fixed random seed for reproducibility.
4. Models: Linear Regression baseline and Random Forest Regressor.
5. Evaluation: MAE, RMSE, and R².
6. Diagnostics: actual-vs-predicted chart and residual plot for the best-scoring model.
7. Interpretation: Linear Regression coefficients and Random Forest feature importance.

## Interpreting the metrics

- **MAE:** average absolute prediction error; lower is better.
- **RMSE:** error metric that gives greater weight to large errors; lower is better.
- **R²:** relative fit compared with predicting the test-set mean; generally, higher is better.

Metrics can change if the data or train/test split changes. Do not make up results in the README or LinkedIn post; run the notebook and report the observed values.

## Limitations

This is a small educational dataset using only three advertising channels. Correlation, regression coefficients, and feature importance are not proof of causation. Real-world sales can also depend on price, seasonality, product availability, competitors, promotions, and other factors missing from this dataset.

## Task checklist coverage

- [x] Load dataset and inspect data quality / summary statistics.
- [x] Pairplot and individual scatter plots for Sales vs. TV, Radio, and Newspaper.
- [x] Correlation heatmap.
- [x] Train/test split.
- [x] Linear Regression baseline.
- [x] Additional regression model (Random Forest Regressor).
- [x] MAE, RMSE, and R² evaluation.
- [x] Residual plot for the selected best model.
- [x] Channel interpretation via coefficients and feature importance.
- [x] Commented code with markdown explanations in the notebook.
