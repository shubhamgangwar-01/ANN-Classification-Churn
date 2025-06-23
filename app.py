import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

## Loading the trained model
model = load_model('/Users/shubhamgangwar/Documents/Python/AI_ML Krish Nayak/AI_ML Krish/Machine Learning/Deep Learning/ANN Project/model.h5')
#print('MODEL Loaded Succesfully')

## Load  the encoders and scaler

with open('/Users/shubhamgangwar/Documents/Python/AI_ML Krish Nayak/AI_ML Krish/Machine Learning/Deep Learning/ANN Project/label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('/Users/shubhamgangwar/Documents/Python/AI_ML Krish Nayak/AI_ML Krish/Machine Learning/Deep Learning/ANN Project/onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('/Users/shubhamgangwar/Documents/Python/AI_ML Krish Nayak/AI_ML Krish/Machine Learning/Deep Learning/ANN Project/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

##streamlit app
st.title('Customer Churn Prediction')

##User Input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0,10)
num_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])


## Preparing the input data
input_data = pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary': [estimated_salary]
})

## OneHot Encoded 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

## Combine the encoded data in the input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

## Scale the input data
input_data_scaled = scaler.transform(input_data)

## Predict Churn
prediction = model.predict(input_data_scaled)
pred_probablily = prediction[0][0]

st.write(f'Churn Probability: {pred_probablily:.2f}')

if pred_probablily > 0.5:
    st.write('The customer is likely to churn')
else:
    st.write('The customer is not likely to churn')








