import streamlit as st
import pandas as pd
import joblib

# --- Page Config ---
st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺")

# --- Load the Pipeline ---
@st.cache_resource # This keeps the model in memory so the app stays fast
def load_model():
    return joblib.load('patients_pipeline.pkl')

pipeline = load_model()

# --- App Header ---
st.title("🩺 Diabetes Health Screening Tool")
st.markdown("""
This tool uses a **Logistic Regression** model to predict the risk of diabetes 
based on patient health metrics.
*Goal: High Sensitivity (Recall) to ensure no potential cases are missed.*
""")

st.divider()

# --- Input Form ---
st.subheader("Patient Health Metrics")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2)
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=0.0, value=120.0)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, value=70.0)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0.0, value=20.0)
    insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0.0, value=80.0)

with col2:
    bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.5)
    age = st.number_input("Age", min_value=21, max_value=120, value=30)
    
    bmi_cat = st.selectbox("BMI Category", ["Underweight", "Normal", "Overweight", "Obese"])
    blood_type = st.selectbox("Blood Type", ["A", "B", "AB", "O"])

# --- Logic: Prepare Features for the Model ---
if st.button("Predict Diabetes Risk"):
    # 1. Map BMI Category (Ordinal)
    bmi_mapping = {"Underweight": 0.0, "Normal": 1.0, "Overweight": 2.0, "Obese": 3.0}
    bmi_encoded = bmi_mapping[bmi_cat]

    # 2. Map Blood Type (One-Hot dummy variables)
    # Our model used A as the base (all 0s)
    bt_B = 1 if blood_type == "B" else 0
    bt_AB = 1 if blood_type == "AB" else 0
    bt_O = 1 if blood_type == "O" else 0

    # 3. Calculate Ratio
    ratio = insulin / (glucose + 1e-6)

    # 4. Create DataFrame (Order MUST match exactly what model expects)
    input_data = pd.DataFrame([{
        'pregnancies': pregnancies,
        'glucose': glucose,
        'blood_pressure': blood_pressure,
        'skin_thickness': skin_thickness,
        'insulin': insulin,
        'bmi': bmi,
        'diabetes_pedigree_function': dpf,
        'age': age,
        'bmi_cat_encoded': bmi_encoded,
        'bt_AB': bt_AB,
        'bt_B': bt_B,
        'bt_O': bt_O,
        'insulin_glucose_ratio': ratio
    }])

    # 5. Get Prediction
    prob = pipeline.predict_proba(input_data)[:, 1][0]
    
    # We use your custom threshold (0.42) to maintain high Recall
    threshold = 0.42
    prediction = 1 if prob >= threshold else 0

    # --- Display Results ---
    st.divider()
    if prediction == 1:
        st.error(f"### Result: High Risk of Diabetes")
        st.write(f"Confidence Level: **{prob*100:.2f}%**")
        st.warning("⚠️ Recommendation: Patient should consult a specialist for further clinical testing.")
    else:
        st.success(f"### Result: Low Risk of Diabetes")
        st.write(f"Confidence Level: **{(1-prob)*100:.2f}%** (Confidence of Healthy)")
        st.info("ℹ️ Recommendation: Continue regular health check-ups.")
