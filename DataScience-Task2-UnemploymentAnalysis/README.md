# Unemployment Analysis with Python — Oasis Infobyte Task 2

## Overview
Exploratory data analysis of unemployment in India, with regional and monthly comparisons and a descriptive comparison of the pre-COVID period with March 2020 onward. This project is built to match the Data Science Task 2 checklist in the Oasis Infobyte internship task list.

## Tools
- Python 3
- Jupyter Notebook
- pandas and NumPy
- Matplotlib and Seaborn

## Dataset
Download the **Unemployment in India** dataset from [Kaggle](https://www.kaggle.com/datasets/gokulrajkmv/unemployment-in-india), extract it, and place `Unemployment in India.csv` in this folder next to the notebook. If your CSV has a different filename, update `DATA_PATH` in the notebook. Column matching supports common variants of the dataset headers.

The raw dataset is not bundled here. If you decide to commit the CSV, confirm its source terms first.

## Run the project
1. Install Python 3 and Jupyter.
2. Open a terminal in this directory and run:

   ```bash
   pip install -r requirements.txt
   jupyter notebook Unemployment_Analysis_Task2.ipynb
   ```

3. Open the notebook and choose **Kernel → Restart Kernel and Run All Cells**.
4. Review the calculated observations, complete the Final Findings section in your own words, and verify that all cells run successfully.

## Outputs
The notebook saves five chart images to `outputs/` when it is run:

- Monthly unemployment trend
- Three-region time series
- Top 10 regions by average unemployment rate
- Correlation heatmap
- Pre-COVID vs COVID-period comparison

## Important interpretation note
The notebook defines pre-COVID as dates before 1 March 2020 and the COVID period as dates from 1 March 2020 onward. The commonly used dataset ends in 2020, so this is not a long-term post-pandemic analysis. The plotted averages are descriptive and unweighted across available rows; correlation does not establish causation.
