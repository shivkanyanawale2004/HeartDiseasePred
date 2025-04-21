import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import firebase_admin
from firebase_admin import credentials, firestore

# Firebase Initialization
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase_key.json")
    firebase_admin.initialize_app(cred)
db = firestore.client()

# Title
st.title('Heart Disease Prediction')

# Load dataset
heart_data = pd.read_csv('dataset.csv')

# Data preprocessing and model training
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Model training
model = LogisticRegression(max_iter=5000)
model.fit(X_train, Y_train)

# Model evaluation
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
st.write(f'Training Accuracy: {training_data_accuracy:.2f}')

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
st.write(f'Test Accuracy: {test_data_accuracy:.2f}')

# User input
st.subheader('Enter Values for Prediction')
user_input = []
input_dict = {}
for col in X.columns:
    val = st.number_input(f"Enter {col}", value=float(0))
    user_input.append(val)
    input_dict[col] = val

# Make prediction
if st.button('Predict'):
    try:
        user_input_array = np.asarray(user_input).reshape(1, -1)
        prediction = model.predict(user_input_array)
        result = 'Has Heart Disease' if prediction[0] == 1 else 'No Heart Disease'

        st.success(f'The person {result}.')

        # Store result in Firebase
        input_dict["prediction"] = result
        db.collection("heart_disease_predictions").add(input_dict)

    except Exception as e:
        st.error(f"Error in prediction: {e}")
