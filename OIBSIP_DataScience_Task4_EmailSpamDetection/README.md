# Data Science Task 4 — Email Spam Detection with Machine Learning

## Objective
Classify messages as spam or legitimate (ham) using NLP and supervised machine learning.

## Tools
- Python
- pandas
- scikit-learn (TF-IDF, Multinomial Naive Bayes, Logistic Regression, metrics)
- matplotlib and seaborn
- Jupyter Notebook

## Dataset
The task brief suggests the SMS Spam Collection dataset from Kaggle or the UCI Machine Learning Repository. Download the dataset, extract it, and place either `spam.csv` or `SMSSpamCollection` next to the notebook. If it has another filename, edit `DATA_PATH` in the notebook. The UCI file is normally tab-separated; common Kaggle copies are CSV.

> Dataset note: the recommended SMS Spam Collection consists of text messages, not full email bodies. State this limitation clearly. If your evaluator requires true email data, use an email dataset and adapt the loader accordingly.

## How to run
1. Install Python 3 and Jupyter Notebook/JupyterLab.
2. Install dependencies: `pip install -r requirements.txt`.
3. Download and extract the dataset as described above.
4. Open `Email_Spam_Detection_Task4.ipynb`.
5. Set `DATA_PATH` if needed, then run the notebook from top to bottom.
6. Use the actual metrics and plots produced by your run to write the final findings section. Do not copy guessed scores.

## What the notebook includes
- Data loading and cleaning
- Spam/ham count and percentage summary
- EDA plots
- Lowercasing, punctuation cleanup, stopword removal, and TF-IDF
- Stratified train/test split
- Multinomial Naive Bayes and Logistic Regression
- Accuracy, spam precision, spam recall, F1-score, classification reports, confusion matrices
- Best model selection by spam F1-score
- Custom-message predictions 
- Optional word clouds

## Folder suggestion for OIBSIP
`OIBSIP/DataScience-Task4-EmailSpamDetection/`

Commit the notebook, this README, `requirements.txt`, and (after running) a few output screenshots. Do not upload the dataset unless its license and sharing terms permit it.
