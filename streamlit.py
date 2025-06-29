import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("🚢 Titanic Survival Prediction")
st.write("This app predicts if a passenger would survive the Titanic disaster based on selected features.")

# Load dataset automatically
@st.cache_data
def load_data():
    df = pd.read_csv(""C:\Users\HP\Desktop\DATA SCIENCE\Assignments\Logistic Regression\STREAMLIT\Titanic_train.csv"")  # Make sure this file is in the same folder
    return df

train_df = load_data()

# Preprocessing
train_df = train_df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']].dropna()
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})

X_train = train_df[['Pclass', 'Sex', 'Age', 'Fare']]
y_train = train_df['Survived']

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)
st.success("Model is preloaded and trained successfully!")

# Sidebar for user to simulate test passenger
st.sidebar.header("Enter Passenger Details")
pclass = st.sidebar.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.sidebar.radio("Sex", ["male", "female"])
age = st.sidebar.slider("Age", 1, 80, 25)
fare = st.sidebar.slider("Fare", 0.0, 600.0, 50.0)

if st.sidebar.button("Predict Survival"):
    input_df = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [1 if sex == 'female' else 0],
        'Age': [age],
        'Fare': [fare]
    })
    prediction = model.predict(input_df)[0]
    result = "✅ Survived" if prediction == 1 else "❌ Did not survive"
    st.subheader("Prediction Result")
    st.write(f"Based on the input, the passenger would have: **{result}**")
