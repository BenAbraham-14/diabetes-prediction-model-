import joblib
import pandas as pd
import streamlit as st

# Set page config
st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")

st.title("🩺 Early-Stage Diabetes Risk Predictor")
st.write("Fill out the symptoms below to check the risk assessment.")


# Load trained pipeline
@st.cache_resource
def load_model():
    artifact = joblib.load("model.joblib")
    return artifact["model"], artifact["features"]


model, features = load_model()

# Form layout
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age", min_value=1, max_value=120, value=45, step=1
        )
        gender = st.selectbox("Gender", ["Male", "Female"])
        weight_loss = st.selectbox("Sudden Weight Loss", ["No", "Yes"])
        weakness = st.selectbox("Weakness", ["No", "Yes"])
        polyphagia = st.selectbox("Polyphagia (Excessive Hunger)", ["No", "Yes"])
        thrush = st.selectbox("Genital Thrush", ["No", "Yes"])
        visual_blurring = st.selectbox("Visual Blurring", ["No", "Yes"])

    with col2:
        itching = st.selectbox("Itching", ["No", "Yes"])
        irritability = st.selectbox("Irritability", ["No", "Yes"])
        delayed_healing = st.selectbox("Delayed Healing", ["No", "Yes"])
        paresis = st.selectbox("Partial Paresis", ["No", "Yes"])
        muscle_stiffness = st.selectbox("Muscle Stiffness", ["No", "Yes"])
        alopecia = st.selectbox("Alopecia (Hair Loss)", ["No", "Yes"])
        obesity = st.selectbox("Obesity", ["No", "Yes"])

    submitted = st.form_submit_button("Predict Risk")

if submitted:
    # Mapping
    binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}

    input_data = {
        "Age": age,
        "Gender": binary_map[gender],
        "sudden weight loss": binary_map[weight_loss],
        "weakness": binary_map[weakness],
        "Polyphagia": binary_map[polyphagia],
        "Genital thrush": binary_map[thrush],
        "visual blurring": binary_map[visual_blurring],
        "Itching": binary_map[itching],
        "Irritability": binary_map[irritability],
        "delayed healing": binary_map[delayed_healing],
        "partial paresis": binary_map[paresis],
        "muscle stiffness": binary_map[muscle_stiffness],
        "Alopecia": binary_map[alopecia],
        "Obesity": binary_map[obesity],
    }

    input_df = pd.DataFrame([input_data])[features]

    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.subheader("Result:")
    if prediction == 1:
        st.error(
            f"⚠️ **High Risk of Diabetes** (Estimated Probability: {prob * 100:.1f}%)"
        )
    else:
        st.success(
            f"✅ **Low Risk of Diabetes** (Estimated Probability: {prob * 100:.1f}%)"
        )
