import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="Health Risk Prediction App",
    page_icon="🩺",
    layout="wide"
)

# Load the trained SVC model
@st.cache_resource
def load_model():
    with open("svmc.pkl", "rb") as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading 'model.pkl': {e}")
    st.stop()

# App Header
st.title("🩺 Health Risk & Diabetes Prediction App")
st.markdown("Please fill in the patient details below to get the prediction from the SVC model.")
st.markdown("---")

# Create input form layout using columns
with st.form("prediction_form"):
    st.subheader("Patient Demographic & Lifestyle Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        city = st.text_input("City", value="New York")
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
        family_history_diabetes = st.selectbox("Family History of Diabetes", ["No", "Yes"])
        
    with col2:
        physical_activity_level = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])
        diet_type = st.selectbox("Diet Type", ["Balanced", "Vegetarian", "Vegan", "Keto", "Other"])
        smoking_status = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
        alcohol_consumption = st.selectbox("Alcohol Consumption", ["None", "Low", "Moderate", "High"])
        hours_sleep_per_night = st.number_input("Hours Sleep Per Night", min_value=1.0, max_value=15.0, value=7.0)
        stress_level = st.slider("Stress Level (1-10)", min_value=1, max_value=10, value=5)
        
    with col3:
        fasting_blood_sugar = st.number_input("Fasting Blood Sugar", min_value=50.0, max_value=300.0, value=100.0)
        hba1c_level = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.5)
        blood_pressure_systolic = st.number_input("Blood Pressure (Systolic)", min_value=70, max_value=200, value=120)
        blood_pressure_diastolic = st.number_input("Blood Pressure (Diastolic)", min_value=40, max_value=120, value=80)
        waist_circumference_cm = st.number_input("Waist Circumference (cm)", min_value=40.0, max_value=200.0, value=85.0)
        income_bracket = st.selectbox("Income Bracket", ["Low", "Medium", "High"])

    submitted = st.form_submit_button("Predict", use_container_width=True)

if submitted:
    # Mappings for categorical inputs
    gender_map = {"Male": 0, "Female": 1, "Other": 2}
    activity_map = {"Low": 0, "Moderate": 1, "High": 2}
    diet_map = {"Balanced": 0, "Vegetarian": 1, "Vegan": 2, "Keto": 3, "Other": 4}
    smoking_map = {"Never": 0, "Former": 1, "Current": 2}
    alcohol_map = {"None": 0, "Low": 1, "Moderate": 2, "High": 3}
    income_map = {"Low": 0, "Medium": 1, "High": 2}
    
    city_code = abs(hash(city)) % 50 
    
    # Construct input dataframe matching model features
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [gender_map.get(gender, 0)],
        'city': [city_code],
        'bmi': [bmi],
        'family_history_diabetes': [1 if family_history_diabetes == "Yes" else 0],
        'physical_activity_level': [activity_map.get(physical_activity_level, 0)],
        'diet_type': [diet_map.get(diet_type, 0)],
        'smoking_status': [smoking_map.get(smoking_status, 0)],
        'alcohol_consumption': [alcohol_map.get(alcohol_consumption, 0)],
        'hours_sleep_per_night': [hours_sleep_per_night],
        'stress_level': [stress_level],
        'fasting_blood_sugar': [fasting_blood_sugar],
        'hba1c_level': [hba1c_level],
        'blood_pressure_systolic': [blood_pressure_systolic],
        'blood_pressure_diastolic': [blood_pressure_diastolic],
        'waist_circumference_cm': [waist_circumference_cm],
        'income_bracket': [income_map.get(income_bracket, 0)]
    })

    try:
        st.markdown("---")
        st.subheader("Prediction Results")
        
        # Check if the model supports probability calculation
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            classes = model.classes_
            
            st.write("### Confidence Breakdown:")
            for cls, prob in zip(classes, probabilities):
                percentage = prob * 100
                st.progress(float(prob), text=f"**{cls}**: {percentage:.2f}%")
        else:
            # Fallback if model was trained without probability=True
            prediction = model.predict(input_data)
            st.success(f"The model prediction is: **{prediction[0]}**")
            st.info("Tip: To display percentages, train your model using `SVC(probability=True)`.")
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")
