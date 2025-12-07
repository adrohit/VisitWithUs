import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="adrohit/VisitWithUs", 
                             filename="best_machine_failure_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Wellness Tourism Package Prediction
st.title("Wellness Tourism Package Prediction App")
st.write("""
This application predicts whether a customer will purchase the newly introduced Wellness Tourism Package.
""")

# User input
Type = st.selectbox("Machine Type", ["H", "L", "M"])
air_temp = st.number_input("Air Temperature (K)", min_value=250.0, max_value=400.0, value=298.0, step=0.1)
process_temp = st.number_input("Process Temperature (K)", min_value=250.0, max_value=500.0, value=324.0, step=0.1)
rot_speed = st.number_input("Rotational Speed (RPM)", min_value=0, max_value=3000, value=1400)
torque = st.number_input("Torque (Nm)", min_value=0.0, max_value=100.0, value=40.0, step=0.1)
tool_wear = st.number_input("Tool Wear (min)", min_value=0, max_value=300, value=10)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Air_temperature': air_temp,
    'Process_temperature': process_temp,
    'Rotational_speed': rot_speed,
    'Torque': torque,
    'Tool_wear': tool_wear,
    'Type': Type
}])

# --- Categorical fields ---
TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
Occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
Gender = st.selectbox("Gender", ["Male", "Female"])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married","Unmarried", "Divorced"])
Passport = st.selectbox("Passport", [0, 1])
OwnCar = st.selectbox("Own Car", [0, 1])
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "VP", "AVP"])
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
 
# --- Numerical fields ---
Age = st.number_input("Age", min_value=18, max_value=80, value=30)
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
PreferredPropertyStar = st.number_input("Preferred Property Star", min_value=1, max_value=5, value=3)
NumberOfTrips = st.number_input("Number of Trips per Year", min_value=0, max_value=30, value=3)
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=0)
MonthlyIncome = st.number_input("Monthly Income", min_value=0, max_value=500000, value=40000)
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
NumberOfFollowups = st.number_input("Number of Followups", min_value=0, max_value=50, value=3)
DurationOfPitch = st.number_input("Duration of Pitch (min)", min_value=0, max_value=60, value=10)

# --- Assemble into DataFrame ---
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'ProductPitched': ProductPitched,
    'NumberOfFollowups': NumberOfFollowups,
    'DurationOfPitch': DurationOfPitch
}])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    result = "Product Taken" if prediction == 1 else "Not Taken"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
