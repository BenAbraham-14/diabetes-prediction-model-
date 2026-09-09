import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="DiaPredict | Clinical Risk Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f8fafc;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
        margin-bottom: 20px;
    }
    .stButton>button {
        width: 100%;
        background-color: #0284c7;
        color: white;
        font-weight: 600;
        height: 3em;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #0369a1;
        color: white;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# Load Model Artifact
@st.cache_resource
def load_model():
    data = joblib.load("model.joblib")
    model_obj = data.get("model", data.get("pipeline"))
    return model_obj, data["features"]


model, features = load_model()

# Sidebar: Context & Information
with st.sidebar:
    st.image(
        "https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?w=500&auto=format&fit=crop&q=60",
        use_container_width=True,
    )
    st.title("DiaPredict AI")
    st.caption("Machine Learning Risk Screening Tool")
    st.info(
        "This tool uses an ensemble Random Forest trained on clinical cohort data to flag early indicators of diabetes risk."
    )
    st.markdown("---")
    st.markdown(
        """
    **Key Warning Indicators:**
    * **Polyuria:** Frequent urination
    * **Polydipsia:** Unquenchable thirst
    * **Polyphagia:** Excessive hunger
    * **Rapid Weight Loss**
    """
    )
    st.markdown("---")
    st.caption("⚠️ **Notice:** For preliminary screening only. Not a medical diagnosis.")

# Main Header
st.title("🩺 Early-Stage Diabetes Risk Evaluator")
st.write(
    "Select the patient's symptoms below to generate a probabilistic risk assessment."
)

# Input Layout using Columns
col_form, col_spacer, col_preview = st.columns([1.5, 0.1, 1.2])

with col_form:
    with st.form("clinical_form"):
        st.subheader("1. Demographics")
        d1, d2 = st.columns(2)
        with d1:
            age = st.slider(
                "Patient Age", min_value=15, max_value=90, value=45, step=1
            )
        with d2:
            gender = st.radio(
                "Biological Sex",
                ["Male", "Female"],
                horizontal=True,
                help="Biological sex influences symptom presentation rates.",
            )

        st.subheader("2. Hallmark Symptoms (High Priority)")
        st.caption("Presence of these symptoms strongly shifts the risk curve.")
        h1, h2 = st.columns(2)
        with h1:
            polyuria = st.selectbox(
                "Polyuria (Excessive Urination)", ["No", "Yes"]
            )
            weight_loss = st.selectbox("Sudden Weight Loss", ["No", "Yes"])
        with h2:
            polydipsia = st.selectbox(
                "Polydipsia (Excessive Thirst)", ["No", "Yes"]
            )
            polyphagia = st.selectbox(
                "Polyphagia (Excessive Hunger)", ["No", "Yes"]
            )

        st.subheader("3. Secondary Clinical Indicators")
        with st.expander("Expand Secondary Symptoms & Conditions", expanded=True):
            s1, s2 = st.columns(2)
            with s1:
                weakness = st.selectbox("Weakness / Fatigue", ["No", "Yes"])
                visual_blurring = st.selectbox("Visual Blurring", ["No", "Yes"])
                genital_thrush = st.selectbox("Genital Thrush", ["No", "Yes"])
                itching = st.selectbox("Persistent Itching", ["No", "Yes"])
            with s2:
                irritability = st.selectbox("Irritability", ["No", "Yes"])
                delayed_healing = st.selectbox(
                    "Delayed Wound Healing", ["No", "Yes"]
                )
                partial_paresis = st.selectbox(
                    "Partial Paresis (Muscle Weakness)", ["No", "Yes"]
                )
                muscle_stiffness = st.selectbox(
                    "Muscle Stiffness", ["No", "Yes"]
                )
                alopecia = st.selectbox("Alopecia (Hair Loss)", ["No", "Yes"])
                obesity = st.selectbox("Obesity", ["No", "Yes"])

        submitted = st.form_submit_button("Run Risk Assessment")

# Processing & Results Section
with col_preview:
    st.subheader("Clinical Assessment")

    if submitted:
        # Format payload
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

        # Predict
        input_df = pd.DataFrame([input_data])[features]
        risk_probability = model.predict_proba(input_df)[0][1]
        risk_percent = round(risk_probability * 100, 1)

        # Risk Tier Logic
        if risk_percent < 30:
            status_text = "Low Risk"
            alert_type = "success"
            badge_color = "#16a34a"
            rec_text = "No prominent risk signatures detected. Continue routine checkups and balanced lifestyle habits."
        elif risk_percent < 60:
            status_text = "Moderate / Borderline Risk"
            alert_type = "warning"
            badge_color = "#ca8a04"
            rec_text = "Several secondary indicators are present. A preventive HbA1c screening is recommended."
        else:
            status_text = "High Risk Detected"
            alert_type = "error"
            badge_color = "#dc2626"
            rec_text = "Key clinical hallmarks detected. Prioritize formal fasting blood glucose and diagnostic lab work."

        # Risk Visual Card
        st.markdown(
            f"""
            <div style="background-color: white; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0; text-align: center;">
                <p style="margin: 0; color: #64748b; font-size: 0.95em; text-transform: uppercase; font-weight: 600;">Computed Probability</p>
                <h1 style="margin: 5px 0 0 0; color: {badge_color}; font-size: 3.2em;">{risk_percent}%</h1>
                <span style="background-color: {badge_color}20; color: {badge_color}; padding: 4px 12px; border-radius: 20px; font-weight: 700; font-size: 0.85em;">
                    {status_text}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        st.progress(risk_percent / 100)

        # Recommendation box
        if alert_type == "success":
            st.success(f"**Recommendation:** {rec_text}")
        elif alert_type == "warning":
            st.warning(f"**Recommendation:** {rec_text}")
        else:
            st.error(f"**Recommendation:** {rec_text}")

        # Active Symptoms Breakdown
        st.markdown("#### Reported Positive Symptoms")
        active_symptoms = [
            k for k, v in input_data.items() if v == 1 and k != "Gender"
        ]

        if active_symptoms:
            for symptom in active_symptoms:
                is_hallmark = symptom in [
                    "Polyuria",
                    "Polydipsia",
                    "sudden weight loss",
                    "Polyphagia",
                ]
                badge = "🔴 High-Weight Indicator" if is_hallmark else "⚪ Factor"
                st.markdown(f"- **{symptom.capitalize()}** ({badge})")
        else:
            st.info("No positive symptoms selected.")

    else:
        st.info(
            "👈 Enter patient metrics and click **Run Risk Assessment** to display findings."
        )  )
