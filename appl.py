import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle5 as pickle

file1 = open(r'D:\\DATA_SCIENCE\\Python\\Assignment\\Logistic Regression\\laptop.pkl', 'rb')
rf = pickle.load(file1)
file1.close()

data = pd.read_csv("D:\\DATA_SCIENCE\\Python\\Assignment\\Logistic Regression\\train_df.csv")

st.title("Titanic Survival Prediction")

# Pclass
Pclass = st.selectbox('Select Pclass', [1, 2, 3])

# Age
Age = st.number_input('Enter Age of the passenger', min_value=0, max_value=100)

# SibSp (siblings/spouses aboard)
SibSp = st.selectbox('Select SibSp', [0, 1, 2, 3, 4, 5, 8])

# Parch (parents/children aboard)
Parch = st.selectbox('Select Parch', [0, 1, 2, 3, 4, 5, 6, 9])

# Gender (Male or Female)
Gender = st.selectbox('Select Gender', ['male', 'female'])
# Mapping Gender to numerical values
gender_mapping = {'male': 1, 'female': 0}
Gender = gender_mapping[Gender]

# Fare (Medium / High / Very High)
Fare = st.number_input('Enter Fare', min_value=0, max_value=550)

# Embarked (Embarkation point)
Embarked = st.selectbox('Select Embarked', ['C', 'Q', 'S'])
# Mapping Embarked to numerical values
embarked_mapping = {'C': 3, 'Q': 2, 'S': 1}
Embarked = embarked_mapping[Embarked]



# Create a DataFrame for the user input
input_data = pd.DataFrame({
    'Pclass': [Pclass],
    'Age': [Age],
    'SibSp': [SibSp],
    'Parch': [Parch],
    'Sex': [Gender],
    'Fare': [Fare],
    'Embarked': [Embarked]
})


# Prediction based on the input data
if st.button('Predict Survival'):
    prediction = rf.predict(input_data)
    survival_status = "Survived" if prediction == 1 else "Did not survive"
    st.write(f"The passenger is predicted to have: {survival_status}")
