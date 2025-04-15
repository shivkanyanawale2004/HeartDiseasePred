import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate('serviceAccountKey.json')
    firebase_admin.initialize_app(cred)

db = firestore.client()

# App Title
st.title('Heart Disease Prediction')

# Load dataset
heart_data = pd.read_csv('dataset.csv')

# Data preprocessing
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train the model
model = LogisticRegression(max_iter=5000)
model.fit(X_train, Y_train)

# Accuracy Display
st.write(f"Training Accuracy: {accuracy_score(Y_train, model.predict(X_train)):.2f}")
st.write(f"Test Accuracy: {accuracy_score(Y_test, model.predict(X_test)):.2f}")

# Input section
st.subheader('Enter Values for Prediction')
user_input = []
for col in X.columns:
    val = st.number_input(f"{col}", value=0.0)
    user_input.append(val)

# Predict and store in Firestore
if st.button('Predict'):
    try:
        input_array = np.asarray(user_input).reshape(1, -1)
        prediction = model.predict(input_array)
        result = 'The person has heart disease.' if prediction[0] == 1 else 'The person does not have heart disease.'
        st.success(result)

        # Firestore Save
        timestamp = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
        doc_name = f"heart_{timestamp}"

        db.collection('heart_predictions').document(doc_name).set({
            'input_data': dict(zip(X.columns, user_input)),
            'prediction': int(prediction[0]),
            'result_text': result,
            'timestamp': timestamp
        })

        st.info(f"Prediction saved as: {doc_name}")
    except Exception as e:
        st.error(f"Prediction error: {e}")