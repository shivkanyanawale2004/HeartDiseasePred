import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Title
st.title('Heart Disease Prediction')

# Load dataset directly
heart_data = pd.read_csv(r'C:\Users\userm\OneDrive\Desktop\HeartDisease\dataset.csv')


# Data preprocessing and model training
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Model training
model = LogisticRegression(max_iter=5000)  # Increase iterations for convergence
model.fit(X_train, Y_train)

# Model evaluation
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
st.write(f'Training Accuracy: {training_data_accuracy:.2f}')

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
st.write(f'Test Accuracy: {test_data_accuracy:.2f}')

# User input for prediction
st.subheader('Enter Values for Prediction')
user_input = []
for col in X.columns:
    value = st.number_input(f"Enter {col}", value=float(0))
    user_input.append(value)

# Make prediction
if st.button('Predict'):
    try:
        user_input_array = np.asarray(user_input).reshape(1, -1)
        prediction = model.predict(user_input_array)
        
        if prediction[0] == 1:
            st.success('The person has heart disease.')
        else:
            st.success('The person does not have heart disease.')
    except Exception as e:
        st.error(f"Error in prediction: {e}")
