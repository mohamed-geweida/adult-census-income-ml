# Adult Census Income – End-to-End Machine Learning Project

[![scikit-learn](https://img.shields.io/badge/scikit--learn-FA9F1C?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-4C9A2A?style=for-the-badge&logo=seaborn&logoColor=white)](https://seaborn.pydata.org/)
[![Made with Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/uciml/adult-census-income)
[![Purpose: Educational](https://img.shields.io/badge/Purpose-Educational-8A2BE2?style=for-the-badge)](#license)

This project demonstrates a **complete machine learning workflow** using the [Adult Census Income dataset](https://www.kaggle.com/datasets/uciml/adult-census-income). It includes **data exploration, cleaning, feature engineering, clustering (KMeans), dimensionality reduction (PCA), and income prediction (Random Forest)** — all implemented with **scikit-learn Pipelines** for modularity and reproducibility.

---

## Overview

The dataset provides demographic, educational, and occupational attributes for nearly 50,000 individuals, to understand patterns and predict whether an individual's income exceeds \$50K annually.

This project covers:

- **Exploratory Data Analysis (EDA)** to understand the dataset
- **Preprocessing & Feature Engineering** to prepare data for modeling
- **Clustering** using KMeans to identify groups of similar individuals
- **Dimensionality Reduction** using PCA for visualization and improved modeling
- **Classification** using Random Forest to predict income levels

---

## Dataset Details

- **Source:** [Kaggle – Adult Census Income](https://www.kaggle.com/datasets/uciml/adult-census-income)
- **Rows:** \~32,561
- **Features:** 14 (age, education, work hours, marital status, occupation, etc.)
- **Target Variable:** `income` (<=50K or >50K)

---

## Workflow

### 1. Data Exploration

- Loaded and inspected the dataset (15 columns; 32,561 rows per `df.info()`)
- Recognized `?` placeholders indicating missing values
- Reviewed feature distributions and unique value counts

### 2. Data Cleaning

- Replaced `?` with NA and removed incomplete rows in key categorical fields
- Reviewed potential outliers in numerical columns

### 3. Exploratory Data Analysis (EDA)

- Analyzed income distribution
- Visualized categorical and numerical features against income
- Correlation analysis among numerical variables

### 4. Feature Engineering & Preprocessing

- Used **ColumnTransformer** to:

  - Scale numerical features with `StandardScaler`
  - Encode categorical variables with `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`

- Applied **Principal Component Analysis (PCA)**:

  - For clustering: 3 components for visualization (`random_state=42`)
  - For classification: retain 95% variance (`n_components=0.95`)

### 5. Clustering (Unsupervised Learning)

- Pipeline: preprocessing → `PCA(n_components=3, random_state=42)` → `KMeans(n_clusters=2, random_state=42)`
- Visualized clusters on the first two PCA components
- Reported PCA explained variance ratio and silhouette score

### 6. Classification (Supervised Learning)

- Split the dataset into training (80%) and testing (20%) sets (`random_state=42`)
- Built a **Random Forest pipeline**: preprocessing → `PCA(n_components=0.95)` → `RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)`
- Evaluated performance using:

  - Accuracy
  - Confusion matrix
  - Classification report

---

## Results

- **K-Means Clustering:** Separated individuals into two meaningful clusters; reported PCA explained variance ratio and silhouette score. PCA used 3 principal components for clustering/visualization.
- **Random Forest Classification:** Accuracy observed at `0.8466` on the held-out test set.

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
cd adult-census-income-ml

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

   - Explore and clean the dataset
   - Perform clustering with K-Means
   - Train and evaluate the Random Forest classifier

---

## Technologies Used

- Python
- pandas, numpy
- matplotlib, seaborn
- scikit-learn (Pipelines, PCA, KMeans, Random Forest)

---

## License

This project is for educational purposes only.
