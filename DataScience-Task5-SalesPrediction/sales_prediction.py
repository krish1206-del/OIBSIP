"""Oasis Infobyte Data Science Task 5: Sales Prediction Using Python.

Install requirements with: pip install -r requirements.txt
Place Advertising.csv beside this script, or ensure the public dataset URL is reachable.
"""


# ------------------------------------------------------------------------
# Import libraries used for data handling, visualisation, modelling, and evaluation.
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)


# ------------------------------------------------------------------------
DATA_PATH = Path("Advertising.csv")
DATA_URL = "https://raw.githubusercontent.com/nguyen-toan/ISLR/master/dataset/Advertising.csv"

if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded local dataset: {DATA_PATH.resolve()}")
else:
    print("Advertising.csv was not found locally; trying the public dataset URL...")
    try:
        df = pd.read_csv(DATA_URL)
        print("Loaded dataset from the public URL.")
    except Exception as exc:
        raise RuntimeError(
            "Could not load Advertising.csv. Download it from "
            "https://github.com/nguyen-toan/ISLR/blob/master/dataset/Advertising.csv "
            "(use the Raw/Download option) and put it beside this notebook."
        ) from exc

# Clean header whitespace and drop any extra exported CSV index column.
df.columns = [str(col).strip() for col in df.columns]
df = df.loc[:, [str(col).strip() and not str(col).lower().startswith("unnamed") for col in df.columns]]

# Match required columns case-insensitively so common CSV variants work.
column_lookup = {str(col).strip().lower(): col for col in df.columns}
required_lower = ["tv", "radio", "newspaper", "sales"]
missing_columns = [col for col in required_lower if col not in column_lookup]
if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}. Found columns: {list(df.columns)}"
    )

df = df[[column_lookup[col] for col in required_lower]].copy()
df.columns = ["TV", "Radio", "Newspaper", "Sales"]

# Convert columns to numbers; text that cannot be converted becomes NaN and is counted below.
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

rows_loaded = len(df)
missing_by_column = df.isna().sum()
rows_with_missing = int(df.isna().any(axis=1).sum())
df = df.dropna(subset=["TV", "Radio", "Newspaper", "Sales"]).copy()
duplicates_removed = int(df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)

print(f"Rows loaded: {rows_loaded}")
print("Missing values before dropping incomplete rows:")
print(missing_by_column.to_string())
print(f"Rows with one or more missing/invalid required values removed: {rows_with_missing}")
print(f"Exact duplicate rows removed: {duplicates_removed}")
print(f"Rows retained for analysis: {len(df)}")
print("\nFirst five rows:")
print(df.head().to_string(index=False))


# ------------------------------------------------------------------------
print("Dataset shape (rows, columns):", df.shape)
print("\nData types:")
print(df.dtypes.to_string())
print("\nDescriptive statistics:")
print(df.describe().round(2).to_string())


# ------------------------------------------------------------------------
sns.pairplot(df, diag_kind="hist", corner=True, diag_kws={"color": "steelblue"})
plt.suptitle("Advertising Spend and Sales: Pairplot", y=1.02)
plt.show()


# ------------------------------------------------------------------------
for channel in ["TV", "Radio", "Newspaper"]:
    plt.figure(figsize=(7, 4.5))
    sns.scatterplot(data=df, x=channel, y="Sales", alpha=0.8)
    sns.regplot(data=df, x=channel, y="Sales", scatter=False, color="red", line_kws={"linewidth": 2})
    plt.title(f"Sales vs. {channel} Advertising Spend")
    plt.xlabel(f"{channel} Advertising Spend")
    plt.ylabel("Sales")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------------
correlation_matrix = df[["TV", "Radio", "Newspaper", "Sales"]].corr(numeric_only=True)
plt.figure(figsize=(7, 5))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Correlation Matrix: Advertising and Sales")
plt.tight_layout()
plt.show()

print("Correlation with Sales, strongest positive to weakest:")
print(correlation_matrix["Sales"].drop("Sales").sort_values(ascending=False).round(3).to_string())


# ------------------------------------------------------------------------
feature_columns = ["TV", "Radio", "Newspaper"]
X = df[feature_columns]
y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"Training rows: {len(X_train)}")
print(f"Testing rows:  {len(X_test)}")


# ------------------------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1
    )
}

predictions = {}
metric_rows = []

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[model_name] = y_pred
    metric_rows.append({
        "Model": model_name,
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred)
    })

print("Both models trained successfully.")


# ------------------------------------------------------------------------
metrics_df = pd.DataFrame(metric_rows).sort_values(
    by=["R2", "RMSE"], ascending=[False, True]
).reset_index(drop=True)
print(metrics_df.round(4).to_string(index=False))

best_model_name = metrics_df.loc[0, "Model"]
best_model = models[best_model_name]
best_predictions = predictions[best_model_name]

print(f"\nBest model on this test split: {best_model_name}")


# ------------------------------------------------------------------------
plt.figure(figsize=(7, 5))
for model_name, y_pred in predictions.items():
    plt.scatter(y_test, y_pred, alpha=0.7, label=model_name)

line_min = min(float(y_test.min()), min(float(np.min(p)) for p in predictions.values()))
line_max = max(float(y_test.max()), max(float(np.max(p)) for p in predictions.values()))
plt.plot([line_min, line_max], [line_min, line_max], linestyle="--", color="black", label="Perfect prediction")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs. Predicted Sales")
plt.legend()
plt.tight_layout()
plt.show()


# ------------------------------------------------------------------------
residuals = y_test.to_numpy() - best_predictions

plt.figure(figsize=(7, 5))
plt.scatter(best_predictions, residuals, alpha=0.8)
plt.axhline(0, linestyle="--", color="red", linewidth=2)
plt.xlabel("Predicted Sales")
plt.ylabel("Residual (Actual - Predicted)")
plt.title(f"Residual Plot — {best_model_name}")
plt.tight_layout()
plt.show()

print(f"Mean residual: {np.mean(residuals):.4f}")
print("Inspect the residual plot for systematic patterns, not just whether the mean is near zero.")


# ------------------------------------------------------------------------
# Linear Regression coefficient analysis.
coefficient_df = pd.DataFrame({
    "Channel": feature_columns,
    "Coefficient": models["Linear Regression"].coef_
})
coefficient_df["Absolute Coefficient"] = coefficient_df["Coefficient"].abs()
coefficient_df = coefficient_df.sort_values("Absolute Coefficient", ascending=False)
print("Linear Regression coefficients (sorted by absolute size):")
print(coefficient_df.round(4).to_string(index=False))

plt.figure(figsize=(7, 4.5))
coef_plot = coefficient_df.sort_values("Coefficient")
plt.bar(coef_plot["Channel"], coef_plot["Coefficient"])
plt.axhline(0, color="black", linewidth=1)
plt.title("Linear Regression Coefficients by Channel")
plt.ylabel("Coefficient (sales change per spend unit)")
plt.tight_layout()
plt.show()

# Random Forest feature importance analysis.
rf_importance_df = pd.DataFrame({
    "Channel": feature_columns,
    "Feature Importance": models["Random Forest Regressor"].feature_importances_
}).sort_values("Feature Importance", ascending=False)
print("Random Forest feature importance:")
print(rf_importance_df.round(4).to_string(index=False))

plt.figure(figsize=(7, 4.5))
rf_plot = rf_importance_df.sort_values("Feature Importance")
plt.barh(rf_plot["Channel"], rf_plot["Feature Importance"])
plt.title("Random Forest Feature Importance")
plt.xlabel("Relative Importance")
plt.ylabel("Advertising Channel")
plt.tight_layout()
plt.show()

highest_linear = coefficient_df.iloc[0]
highest_rf = rf_importance_df.iloc[0]
print(
    f"\nLargest absolute Linear Regression coefficient: {highest_linear['Channel']} "
    f"({highest_linear['Coefficient']:.4f})."
)
print(
    f"Highest Random Forest feature importance: {highest_rf['Channel']} "
    f"({highest_rf['Feature Importance']:.4f})."
)

if best_model_name == "Random Forest Regressor":
    best_channel = highest_rf["Channel"]
    best_channel_reason = "it has the highest Random Forest feature-importance score"
else:
    best_channel = highest_linear["Channel"]
    best_channel_reason = "it has the largest absolute Linear Regression coefficient"
print(f"Based on the selected model, {best_channel} appears most influential because {best_channel_reason}.")


# ------------------------------------------------------------------------
example_spend = pd.DataFrame([{"TV": 230.0, "Radio": 37.0, "Newspaper": 69.0}])
example_prediction = best_model.predict(example_spend)[0]
print("Example advertising spend:")
print(example_spend.to_string(index=False))
print(f"Predicted sales from {best_model_name}: {example_prediction:.2f}")
