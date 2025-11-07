#  Retail Sales Machine Learning Pipeline (PySpark)

[![Python](https://img.shields.io/badge/Python-3.8+-blue)]()
[![PySpark](https://img.shields.io/badge/PySpark-ML%20Pipeline-orange)]()
[![Status](https://img.shields.io/badge/Status-Active-success)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This project demonstrates an **end-to-end machine learning pipeline using PySpark**, designed for retail sales prediction.  
The goal is to predict the **Total Amount spent per transaction** using customer demographics, product information, and purchase behavior.


---

##  Project Overview

This pipeline performs:

1. **Data Loading & Exploration**
2. **Preprocessing & Feature Engineering**
3. **Categorical Encoding + Feature Scaling**
4. **Model Training (Gradient Boosted Trees)**
5. **Model Evaluation (Train/Test RMSE)**

The script is structured using a PySpark **ML Pipeline**, ensuring modularity and reusability.

---

##  Technologies Used

| Component | Technology |
|----------|------------|
| Language | Python |
| Distributed Processing | Apache Spark (PySpark) |
| Model | Gradient Boosted Tree Regressor |
| Environment | Google Colab |

---

##  File Structure
``` bash
├── retail_sales_pipeline.py # Main script (PySpark pipeline)
├── README.md # Project documentation
└── retail_sales_dataset.csv # Dataset
````

---

##  Features & Transformations

| Stage | Description |
|--------|-------------|
| Data Cleaning | Removes unused ID columns |
| Date Feature Engineering | Extracts Year, Month, DayOfWeek |
| Age Binning | Creates demographic Age Groups |
| Quantity Flag | Adds binary feature for high quantity purchases |
| Categorical Encoding | `StringIndexer` + `OneHotEncoder` |
| Numeric Scaling | `StandardScaler` |
| Model | Gradient Boosted Tree Regression |

The target variable (`Total Amount`) is log-transformed to improve prediction accuracy.

---

##  Model Evaluation

The evaluation metric is **RMSE (Root Mean Squared Error)**.

The script outputs:

- RMSE on training dataset
- RMSE on test dataset

This helps identify potential **overfitting** or **underfitting**.

---

##  Outputs
Train RMSE: 3.411323083770224e-11
Test RMSE: 3.5863328269939725e-11


---

##  How to Run

###  Option 1: Run on Google Colab *(recommended)*

1. Upload the dataset (`retail_sales_dataset.csv`) to Colab
2. Upload `retail_sales_pipeline.py`
3. Run the script — Java + PySpark install commands are already included

### ✅ Option 2: Run locally

```bash
pip install pyspark
python retail_sales_pipeline.py
````
 ## What You Learn From This Project

- Hands-on experience with PySpark ML Pipelines

- Feature engineering and preprocessing on distributed data

- Training and evaluating machine learning models at scale

- Applying best practices for real-world data science projects

  ## License
This project is open source and licensed under the MIT License.
You will find the full license in the LICENSE file in this repository.
