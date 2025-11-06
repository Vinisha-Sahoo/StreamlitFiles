# 🚢 Titanic Survival Prediction using Logistic Regression

This project aims to predict the survival of passengers aboard the Titanic using **Logistic Regression**, a supervised machine learning algorithm suitable for binary classification problems.  
It involves complete steps from data exploration and preprocessing to model training, evaluation, and deployment using **Streamlit**.

---


---

## 🧩 1. Data Exploration (EDA)

- Loaded the Titanic dataset and explored its structure.
- Analyzed feature types, summary statistics, and missing values.
- Visualized data using:
  - **Histograms** to show age, fare, and class distributions.
  - **Count plots** for categorical variables like `Sex`, `Embarked`, and `Pclass`.
  - **Correlation heatmap** to examine relationships between numerical features.

### 🔍 Insights:
- **Women and children** had higher survival rates.
- **1st class passengers** were more likely to survive than those in 2nd or 3rd class.
- **Age** and **Fare** showed moderate correlation with survival.

---

## 🧹 2. Data Preprocessing

Steps performed before model training:

1. **Handling Missing Values**  
   - Replaced missing `Age` values with the median.  
   - Filled missing `Embarked` values with the mode.

2. **Encoding Categorical Variables**  
   - Converted categorical columns (`Sex`, `Embarked`) into numeric format using one-hot encoding.

3. **Feature Scaling (if required)**  
   - Applied normalization for numerical stability in logistic regression.

---

## 🤖 3. Model Building (Logistic Regression)

The **Logistic Regression** model was built using **scikit-learn**.

### Steps:
1. Selected features:
['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

2. Encoded categorical features.
3. Split dataset into training and testing sets (80% - 20%).
4. Trained the Logistic Regression model:
```python
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

