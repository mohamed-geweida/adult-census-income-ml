# Adult Census Income – End-to-End Machine Learning Pipelines

This project demonstrates a **complete machine learning workflow** using the [Adult Census Income dataset](https://www.kaggle.com/datasets/uciml/adult-census-income). It includes **data exploration, cleaning, feature engineering, clustering (KMeans), dimensionality reduction (PCA), and income prediction (Random Forest)** — all implemented with **scikit-learn Pipelines** for modularity and reproducibility.

---

## Overview

The dataset provides demographic, educational, and occupational attributes for nearly 50,000 individuals, to understand patterns and predict whether an individual's income exceeds \$50K annually.

This project covers:

* **Exploratory Data Analysis (EDA)** to understand the dataset
* **Preprocessing & Feature Engineering** to prepare data for modeling
* **Clustering** using KMeans to identify groups of similar individuals
* **Dimensionality Reduction** using PCA for visualization and improved modeling
* **Classification** using Random Forest to predict income levels

---

## Dataset Details

* **Source:** [Kaggle – Adult Census Income](https://www.kaggle.com/datasets/uciml/adult-census-income)
* **Rows:** \~32,561
* **Features:** 14 (age, education, work hours, marital status, occupation, etc.)
* **Target Variable:** `income` (<=50K or >50K)

---

## Workflow

### 1. Data Exploration

* Loaded and inspected the dataset
* Checked for missing values and data types
* Reviewed feature distributions and unique value counts

### 2. Data Cleaning

* Removed missing values in `workclass`, `occupation`, and `native-country`
* Handled outliers in numerical columns

### 3. Exploratory Data Analysis (EDA)

* Analyzed income distribution
* Visualized categorical and numerical features against income
* Correlation analysis among numerical variables

### 4. Feature Engineering & Preprocessing

* Used **ColumnTransformer** to:

  * Scale numerical features with StandardScaler
  * Encode categorical variables with OneHotEncoder
* Applied **Principal Component Analysis (PCA)**:

  * For clustering: 2 components for visualization
  * For classification: 95% variance retained

### 5. Clustering (Unsupervised Learning)

* Applied KMeans clustering with optimal number of clusters (k=2)
* Visualized clusters using PCA components
* Analyzed clusters in relation to demographic patterns

### 6. Classification (Supervised Learning)

* Split the dataset into training (80%) and testing (20%) sets
* Built a **Random Forest pipeline** (preprocessing → PCA → classifier)
* Evaluated performance using:

  * Accuracy
  * Confusion matrix
  * Classification report

---

## Results

* **K-Means Clustering:** Separated individuals into two meaningful clusters reflecting income-related patterns.
* **Random Forest Classification:** Achieved reliable prediction performance with strong accuracy and balanced classification metrics.

---

## Project Structure

```
.
├── adult_census_income.ipynb   # Main notebook with pipelines
├── README.md                   # Documentation
├── requirements.txt            # Dependencies
└── adult.csv                   # Dataset
```

---

## Installation

```bash
# Clone repository
git clone https://github.com/mohamed-geweida/adult-census-income-ml.git
cd adult-census-income

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/uciml/adult-census-income) and save it as `adult.csv` in the project folder.
2. Launch the notebook:

```bash
jupyter notebook adult_census_income.ipynb
```

3. Run the cells to:

   * Explore and clean the dataset
   * Perform clustering with K-Means
   * Train and evaluate the Random Forest classifier

---

## Technologies Used

* Python
* pandas, numpy
* matplotlib, seaborn
* scikit-learn (Pipelines, PCA, KMeans, Random Forest)

---

## License

This project is for educational purposes only.

