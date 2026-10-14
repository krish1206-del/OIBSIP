# OASIS Infobyte — Data Science Task 3: Car Price Prediction

## Objective
Build and compare machine-learning regression models that estimate a used car's selling price using features such as brand, age, kilometres driven, fuel type and transmission.

## Tools
- Python
- pandas and NumPy
- Matplotlib and Seaborn
- scikit-learn
- Jupyter Notebook

## Dataset
Use the CarDekho vehicle dataset from Kaggle: https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho

Download the CSV and place it beside the notebook, or edit `DATA_PATH` in the first data-loading cell. The notebook supports common filenames such as `car data.csv` and `CAR DETAILS FROM CAR DEKHO.csv` and standardises common column-name variants.

**Note on price units:** datasets differ. Some files express prices in rupees, while the compact `car data.csv` dataset commonly expresses prices in lakhs. The notebook keeps the unit used by your selected CSV; document the unit when you report results.

## How to run
1. Install Python 3 and Jupyter Notebook, or use Google Colab.
2. Install dependencies: `pip install -r requirements.txt`.
3. Download the CSV and put it in the same folder as the notebook.
4. Open `OIBSIP_DataScience_Task3_CarPricePrediction.ipynb` and run all cells from top to bottom.
5. Read the generated charts and metrics, then complete the findings section with observations from your actual results.

## Workflow
1. Load and inspect the CSV.
2. Clean duplicates, invalid rows, data types and text categories.
3. Engineer `car_age` and `brand` features.
4. Explore selling-price distribution, fuel type, car age and numerical correlations.
5. One-hot encode categorical fields and impute any remaining missing feature values inside a scikit-learn pipeline.
6. Split data into training and test sets (80/20).
7. Train and compare Linear Regression and Random Forest Regressor.
8. Evaluate using MAE, RMSE and R².
9. Plot actual vs predicted values and permutation feature importance.
10. Optionally save the best trained pipeline as `best_car_price_model.joblib`.

## Important interpretation notes
- The model is selected based on the lower test-set RMSE; Random Forest is not guaranteed to perform best.
- `Present_Price` is included as a feature only when the selected dataset has that column.
- This is an educational project, not a live market valuation tool. Historical data, missing features and regional market changes can affect predictions.
