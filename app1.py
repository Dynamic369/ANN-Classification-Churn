import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import tensorflow as tf
import pickle

# Load the trained model
model = tf.keras.models.load_model('regression_model.h5')

#Load the Standard Scaler and encoder

with open('label_encoder_gender1.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('onehot_encoder_geo1.pkl','rb') as file:
    onehot_encoder_geo=pickle.load(file)

with open('scaler1.pkl','rb') as file:
    scaler=pickle.load(file)

#Streamlit app

st.title("Customer Salary Prediction")

#User Input
geography = st.selectbox("Geography",onehot_encoder_geo.categories_[0])
gender=st.selectbox("Gender",label_encoder_gender.classes_)
age = st.slider("Age",18,92)
balance=st.number_input("Balance")
credit_score=st.number_input("Credit Score")
tenure=st.slider('Tenure',1,10)
num_of_products=st.slider('Number of products',1,4)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('Is Active Memeber',[0,1])
exited = st.selectbox('Exited',[0,1])


#Prepare the input data
input_data = pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'Exited':[exited]
})

#One hot encode geography
geo_encoded=onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))


#Combine one hot encoded column with the inout data
input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

#Scale the input data
input_data_scaled= scaler.transform(input_data)

#Prediction Salary
prediction = model.predict(input_data_scaled)
prediction_probability = prediction[0][0]

# st.write(f'Salary Probability;{prediction_probability:.2f}')
st.write(f'Estimated Salary is: {prediction} ')

