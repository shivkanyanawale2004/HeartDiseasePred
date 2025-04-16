import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

# -------------------- Firebase Setup --------------------
if not firebase_admin._apps:
    try:
        # Load Firebase credentials from st.secrets as a dictionary
        firebase_config = dict(st.secrets["firebase"])

        # Convert st.secrets to a credential certificate
        cred = credentials.Certificate(firebase_config)
        firebase_admin.initialize_app(cred)
        db = firestore.client()  # Firebase Firestore client initialization
        st.success("✅ Firebase Initialized Successfully.")
    except Exception as e:
        st.error(f"❌ Error initializing Firebase: {e}")

# -------------------- Title --------------------
st.title('❤️ Heart Disease Prediction App')

# -------------------- Load Dataset --------------------
@st.cache_data
def load_data():
    try:
        heart_data = pd.read_csv('dataset.csv')
        return heart_data
    except FileNotFoundError:
        st.error("❌ Dataset not found. Please check the file path.")
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")

heart_data = load_data()

# -------------------- Data Preprocessing --------------------
if heart_data is not None:
    X = heart_data.drop(columns='target', axis=1)
    Y = heart_data['target']

    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

    # -------------------- Model Training --------------------
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, Y_train)

    # -------------------- Accuracy --------------------
    train_accuracy = accuracy_score(Y_train, model.predict(X_train))
    test_accuracy = accuracy_score(Y_test, model.predict(X_test))
    st.write(f"✅ Training Accuracy: *{train_accuracy:.2f}*")
    st.write(f"✅ Test Accuracy: *{test_accuracy:.2f}*")

    # -------------------- User Input --------------------
    st.subheader("🧾 Enter Patient Details for Prediction")
    user_input = []
    for col in X.columns:
        value = st.number_input(f"{col}", min_value=0.0, value=0.0)
        user_input.append(value)

    # -------------------- Predict Button --------------------
    if st.button('🔍 Predict'):
        try:
            input_array = np.asarray(user_input).reshape(1, -1)
            prediction = model.predict(input_array)

            result = 'has heart disease' if prediction[0] == 1 else 'does NOT have heart disease'
            st.success(f"The person {result}.")

            # -------------------- Save Prediction to Firestore --------------------
            now = datetime.now()
            doc_name = f"heart_{prediction[0]}_{now.strftime('%Y%m%d%H%M%S')}"

            doc_data = {col: val for col, val in zip(X.columns, user_input)}
            doc_data["prediction"] = int(prediction[0])
            doc_data["timestamp"] = now.strftime("%Y-%m-%d %H:%M:%S")

            db.collection("heart_predictions").document(doc_name).set(doc_data)
            st.info(f"✅ Prediction saved to Firestore: {doc_name}")

        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")
