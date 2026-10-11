# Iris Flower Classification

## Objective
Train a machine learning classification model to identify the species of an iris flower (Setosa, Versicolor, or Virginica) based on its physical measurements (sepal length, sepal width, petal length, petal width).

## Tech Stack
* Python
* pandas
* scikit-learn
* matplotlib & seaborn
* Jupyter Notebook

## Project Workflow
1. **Data Loading:** Loaded the Iris dataset directly from `scikit-learn`.
2. **Exploratory Data Analysis (EDA):** Inspected dataset shape, data types, null values, and summary statistics.
3. **Visualization:** Plotted pairplots to inspect feature separability and box plots for feature distributions.
4. **Splitting:** Applied an 80/20 train/test split.
5. **Model Training:** Trained Logistic Regression and Random Forest Classifier models.
6. **Evaluation:** Evaluated models using accuracy, classification reports, and confusion matrices. Logistic Regression was selected as the optimal model due to high performance and interpretability on linearly separable data.

## Repository Structure
* `Iris_Classification.ipynb`: Complete code and notebook outputs.
* `pairplot.png`: Pairplot visualization showing feature distributions.
* `confusion_matrix.png`: Confusion matrix heatmap.
