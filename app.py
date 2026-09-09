import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")

st.title("🩺 Early-Stage Diabetes Risk Predictor")
st.write(
    "Provide patient clinical parameters and symptoms to evaluate diabetes risk."
)


@st.cache_resource
def load_model():
    data = joblib.load("model.joblib")
    model_obj = data.get("model", data.get("pipeline"))
    features_list = data["features"]
    return model_obj, features_list


model, features = load_model()

with st.form("risk_form"):
    st.subheader("Patient Demographics")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input(
            "Age", min_value=1, max_value=120, value=40, step=1
        )
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])

    st.subheader("Key Hallmark Symptoms")
    c1, c2, c3 = st.columns(3)
    with c1:
        polyuria = st.selectbox("Polyuria (Frequent Urination)", ["No", "Yes"])
    with c2:
        polydipsia = st.selectbox("Polydipsia (Excessive Thirst)", ["No", "Yes"])
    with c3:
        weight_loss = st.selectbox("Sudden Weight Loss", ["No", "Yes"])

    st.subheader("Secondary Symptoms")
    s1, s2 = st.columns(2)
    with s1:
        polyphagia = st.selectbox("Polyphagia (Excess Hunger)", ["No", "Yes"])
        weakness = st.selectbox("Weakness", ["No", "Yes"])
        genital_thrush = st.selectbox("Genital Thrush", ["No", "Yes"])
        visual_blurring = st.selectbox("Visual Blurring", ["No", "Yes"])
        itching = st.selectbox("Itching", ["No", "Yes"])
    with s2:
        irritability = st.selectbox("Irritability", ["No", "Yes"])
        delayed_healing = st.selectbox("Delayed Healing", ["No", "Yes"])
        partial_paresis = st.selectbox("Partial Paresis", ["No", "Yes"])
        muscle_stiffness = st.selectbox("Muscle Stiffness", ["No", "Yes"])
        alopecia = st.selectbox("Alopecia (Hair Loss)", ["No", "Yes"])
        obesity = st.selectbox("Obesity", ["No", "Yes"])

    submitted = st.form_submit_button("Evaluate Risk")

if submitted:
    binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}

    input_data = {
        "Age": age,
        "Gender": binary_map[gender],
        "Polyuria": binary_map[polyuria],
        "Polydipsia": binary_map[polydipsia],
        "sudden weight loss": binary_map[weight_loss],
        "weakness": binary_map[weakness],
        "Polyphagia": binary_map[polyphagia],
        "Genital thrush": binary_map[genital_thrush],
        "visual blurring": binary_map[visual_blurring],
        "Itching": binary_map[itching],
        "Irritability": binary_map[irritability],
        "delayed healing": binary_map[delayed_healing],
        "partial paresis": binary_map[partial_paresis],
        "muscle stiffness": binary_map[muscle_stiffness],
        "Alopecia": binary_map[alopecia],
        "Obesity": binary_map[obesity],
    }

    input_df = pd.DataFrame([input_data])[features]
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.divider()
    st.subheader("Diagnostic Assessment")

    if prob >= 0.50:
        st.error(
            f"⚠️ **High Diabetes Risk Detected** (Calculated Risk: {prob * 100:.1f}%)"
        )
    else:
        st.success(
            f"✅ **Low Diabetes Risk** (Calculated Risk: {prob * 100:.1f}%)"
        )