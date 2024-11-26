import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load preprocessing objects
with open('label_enocer_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('onehotencoder.pkl', 'rb') as file:
    onehotencoder = pickle.load(file)

# Streamlit app
st.title('Customer Churn Prediction')

# Collect user inputs
geography = st.selectbox('Geography', onehotencoder.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number Of Products', 1, 4)
has_cr_card = st.selectbox("Has credit Card", [0, 1])
is_active_member = st.selectbox("Is active member", [0, 1])

# Prepare input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],       # Ensure matching feature names
    'Geography': [geography],           # Placeholder for one-hot encoding
    'Gender': [label_encoder_gender.transform([gender])[0]],  # Encode gender
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode Geography
geo_encoded = onehotencoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehotencoder.get_feature_names_out(['Geography']))

# Combine numerical and encoded features
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Ensure feature alignment for the scaler
# Use the feature names the scaler expects
expected_features = scaler.feature_names_in_
input_data = input_data[expected_features]

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Make prediction
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]



st.write(f"the probability is {prediction_proba}")

# Display result
if prediction_proba > 0.5:
    st.error(f'The customer is likely to churn (Probability: {prediction_proba:.2f})')
else:
    st.success(f'The customer is not likely to churn (Probability: {prediction_proba:.2f})')
