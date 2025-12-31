# Solar PV Power Prediction Using Weather Data

## 📌 Project Overview
This project aims to **predict Solar Photovoltaic (PV) activity** using historical weather data collected from Aswan.  
A variety of **feature reduction techniques and machine learning classifiers** are applied to analyze how dimensionality reduction impacts prediction performance.

The project follows a structured **data science pipeline** from preprocessing to model evaluation.

---

## 📂 Dataset Description
- **Dataset:** Aswan weather dataset
- **File:** `AswanData_weatherdata.csv`
- **Index:** Date (converted to `datetime` format)

The dataset contains numerical meteorological attributes relevant to solar energy production.

---

## 🛠️ Technologies & Libraries
- Python  
- NumPy  
- Pandas  
- Matplotlib & Seaborn  
- Scikit-learn  

---

## 🔄 Project Workflow

### 1. Data Preprocessing
- Converted date column to `datetime`
- Extracted:
  - Year
  - Month
  - Day
- Removed redundant or non-informative features
- Checked for missing values and data consistency

---

## 🔍 Feature Reduction Techniques

To improve efficiency and reduce dimensionality, **multiple feature reduction methods** were applied and compared.

### 🔹 1. Manual Feature Selection (5-Feature Model)
A reduced feature set consisting of the **most influential weather attributes** was selected manually.

**Benefits:**
- Lower dimensionality
- Reduced noise
- Improved generalization

---

### 🔹 2. Principal Component Analysis (PCA)
PCA was applied to transform the original features into a smaller set of **orthogonal principal components** that preserve maximum variance.

**Purpose:**
- Remove multicollinearity
- Improve computational efficiency
- Analyze variance contribution

---

### 🔹 3. Linear Discriminant Analysis (LDA – Feature Reduction)
LDA was used as a **supervised dimensionality reduction technique**, maximizing class separability rather than variance.

**Key Characteristics:**
- Uses class labels
- Maximizes between-class variance
- Minimizes within-class variance

---

### 🔹 4. Singular Value Decomposition (SVD)
SVD was applied as a **matrix factorization-based dimensionality reduction technique**, similar in behavior to PCA but computed using singular values.

**Advantages:**
- Stable numerical decomposition
- Effective for high-dimensional data
- Useful when covariance matrix is ill-conditioned

---

## 🤖 Machine Learning Models Used

The following classifiers were trained and evaluated using both **full and reduced feature spaces**:

| Model | Description |
|------|------------|
| Logistic Regression | Linear baseline classifier |
| K-Nearest Neighbors (KNN) | Distance-based classification |
| Decision Tree | Rule-based model |
| Random Forest | Ensemble learning method |
| Linear Discriminant Analysis (LDA) | Linear probabilistic classifier |
| PCA + LDA | PCA followed by LDA classification |
| SVD + Classifier | SVD-based reduced feature model |
| Neural Network (MLP) | Multi-layer perceptron classifier |

---

## 📊 Model Performance Summary

Models were evaluated using:
- Accuracy
- Confusion matrices
- Precision, Recall, F1-score
- Learning curves

### 🔹 Accuracy Comparison (Approximate)

| Model | Accuracy |
|------|---------|
| Logistic Regression | ~80% |
| KNN | ~82% |
| Decision Tree | ~83% |
| Random Forest | ~85% |
| LDA | ~81% |
| PCA + LDA | ~79% |
| SVD-based Model | ~80% |
| Neural Network (MLP) | **~86% (Best)** |

✅ Neural Networks achieved the highest performance  
✅ Feature reduction improved training stability  
✅ LDA outperformed PCA in class separability  
✅ SVD provided competitive results with reduced dimensionality  

---

## 📈 Visualization Outputs
The notebook includes:
- Feature distributions
- Learning curves
- Accuracy comparison plots
- Confusion matrices

These visualizations help evaluate **bias–variance tradeoffs** and model convergence.

---

## ▶️ How to Run the Project
1. Clone or download the repository
2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
