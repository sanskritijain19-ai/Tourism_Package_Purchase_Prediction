import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="sanskritijain27/purchase-model", filename="best_purchase_prediction_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Wellness Tourism Package Prediction
st.title("Wellness Tourism Package Purchase Prediction App")
st.write("This app predicts whether a customer will purchase the newly introduced Wellness Tourism Package.")
st.write("Kindly enter the customer details to check their likelihood of purchasing the package.")

# Collect user input based on tourism dataset features
Age = st.number_input("Age", min_value=18, max_value=100, value=30)
TypeofContact = st.selectbox("Type of Contact", ['Self Enquiry', 'Company Invited'])
CityTier = st.selectbox("City Tier", [1, 2, 3])
Occupation = st.selectbox("Occupation", ['Salaried', 'Small Business', 'Large Business', 'Free Lancer'])
Gender = st.selectbox("Gender", ['Male', 'Female'])
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
ProductPitched = st.selectbox("Product Pitched", ['Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'King'])
NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3)
PreferredPropertyStar = st.number_input("Preferred Property Star (1-5)", min_value=1, max_value=5, value=3)
MaritalStatus = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced', 'Unmarried'])
NumberOfTrips = st.number_input("Number of Trips Annually", min_value=0, max_value=50, value=2)
Passport = st.selectbox("Passport", [0, 1])
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score (1-5)", min_value=1, max_value=5, value=3)
OwnCar = st.selectbox("Own Car", [0, 1])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=0)
Designation = st.selectbox("Designation", ['Manager', 'Executive', 'Senior Manager', 'AVP', 'VP', 'Director'])
MonthlyIncome = st.number_input("Monthly Income", min_value=0.0, value=25000.0)
DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=60, value=10)

# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'ProductPitched': ProductPitched,
    'NumberOfFollowups': NumberOfFollowups,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome,
    'DurationOfPitch': DurationOfPitch
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):    
    prediction_proba = model.predict_proba(input_data)[:, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "purchase the Wellness Tourism Package" if prediction[0] == 1 else "not purchase the Wellness Tourism Package"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
