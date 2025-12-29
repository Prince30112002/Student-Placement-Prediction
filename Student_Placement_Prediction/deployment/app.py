import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("Student_Placement_Prediction/models/placement_model.pkl")

st.set_page_config(page_title="Student Placement Predictor", layout="centered")

st.title("🎓 Student Placement Prediction App")
st.markdown("Predict whether a student will be **Placed or Not Placed**")

st.divider()

# =====================
# User Inputs
# =====================
gender = st.selectbox("Gender", ["M", "F"])
ssc_p = st.slider("SSC Percentage", 40.0, 100.0, 60.0)
hsc_p = st.slider("HSC Percentage", 40.0, 100.0, 60.0)
degree_p = st.slider("Degree Percentage", 40.0, 100.0, 60.0)
etest_p = st.slider("E-test Percentage", 40.0, 100.0, 60.0)
mba_p = st.slider("MBA Percentage", 40.0, 100.0, 60.0)

ssc_b = st.selectbox("SSC Board", ["Central", "Others"])
hsc_b = st.selectbox("HSC Board", ["Central", "Others"])
hsc_s = st.selectbox("HSC Stream", ["Science", "Commerce", "Arts"])
degree_t = st.selectbox("Degree Type", ["Sci&Tech", "Comm&Mgmt", "Arts"])
specialisation = st.selectbox("MBA Specialisation", ["Mkt&HR", "Mkt&Fin"])
workex = st.selectbox("Work Experience", ["Yes", "No"])

# =====================
# Prediction
# =====================
if st.button("🔮 Predict Placement"):
    input_df = pd.DataFrame([{
        "gender": gender,
        "ssc_p": ssc_p,
        "ssc_b": ssc_b,
        "hsc_p": hsc_p,
        "hsc_b": hsc_b,
        "hsc_s": hsc_s,
        "degree_p": degree_p,
        "degree_t": degree_t,
        "workex": workex,
        "etest_p": etest_p,
        "specialisation": specialisation,
        "mba_p": mba_p,
        "salary": 0.0
    }])

    input_encoded = pd.get_dummies(input_df)

    # Align columns with training data
    model_features = model.feature_names_in_
    input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)

    prediction = model.predict(input_encoded)[0]

    st.divider()
    if prediction == 1:
        st.success("🎉 Congratulations! Student is likely to be **PLACED**")
    else:
        st.error("❌ Student is likely to be **NOT PLACED**")
