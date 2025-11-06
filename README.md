🚢 Titanic Survival Prediction

This project predicts the survival probability of passengers aboard the Titanic using a machine learning model, specifically logistic regression.
It involves data exploration, preprocessing, model building, evaluation, and deployment using Streamlit.


1. Data Exploration


Loaded the Titanic dataset and performed exploratory data analysis (EDA).

Examined feature types, summary statistics, and missing values.

Created visualizations such as:

Histograms for numerical features (Age, Fare, etc.)

Count plots for categorical features (Sex, Embarked, Pclass)

Correlation heatmap to identify feature relationships


2. Data Preprocessing


Handled missing values:

Imputed missing Age with median.

Filled missing Embarked with mode.

Encoded categorical variables:

Used one-hot encoding for Sex and Embarked.

Feature scaling was applied when necessary for logistic regression.


3. Model Building
4. 

A Logistic Regression model was built using the scikit-learn library.

Steps followed:

Defined features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked.

Encoded categorical variables.

Split the data into training and testing sets (80–20).

Trained the Logistic Regression model using the training data.



4..Model Evaluation

Metrics used:

Accuracy

Precision

Recall

F1-score

ROC-AUC Score

Visualization:

ROC curve plotted for logistic regression.


5. Interpretation

The coefficients of the Logistic Regression model were analysed to interpret feature importance.

Key Influential Features:

Sex_female → strong positive effect on survival.

Pclass and Age → negative effect (lower class and older age decreased survival odds).

These insights helped understand which passenger characteristics most influenced survival probability.


6. Deployment with Streamlit

A Streamlit web app was developed for interactive prediction.

Features:

Users can input passenger details (e.g., Age, Gender, Pclass, Fare, etc.).

The app predicts whether the passenger would have survived (Survived / Not Survived).

Deployed locally or on Streamlit Community Cloud.


pip install -r requirements.txt
streamlit run app.py
