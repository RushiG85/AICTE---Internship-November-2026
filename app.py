
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# --------------------------------------------------
# Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="NSAP Scheme Prediction",
    page_icon="🏛️",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "nsap_exploratory_model.joblib"

SCHEME_NAMES = {
    "IGNDPS": "Indira Gandhi National Disability Pension Scheme",
    "IGNOAPS": "Indira Gandhi National Old Age Pension Scheme",
    "IGNWPS": "Indira Gandhi National Widow Pension Scheme",
}


# --------------------------------------------------
# Load the trained model
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )
    return joblib.load(MODEL_PATH)


# --------------------------------------------------
# Application interface
# --------------------------------------------------

st.title("🏛️ NSAP Scheme Prediction System")

st.write(
    "An exploratory machine-learning demonstration that "
    "classifies district-level NSAP scheme records using "
    "beneficiary demographic and coverage proportions."
)

st.warning(
    "Research demonstration only: this model uses aggregated "
    "district-level data. It does not determine an individual's "
    "pension eligibility or entitlement."
)

st.divider()

st.subheader("Enter district-level statistics")

with st.form("prediction_form"):

    st.markdown("#### 1. Beneficiary information")

    total = st.number_input(
        "Total beneficiaries",
        min_value=1,
        value=1000,
        step=100,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        male = st.number_input(
            "Male beneficiaries",
            min_value=0,
            value=500,
            step=10,
        )

    with col2:
        female = st.number_input(
            "Female beneficiaries",
            min_value=0,
            value=500,
            step=10,
        )

    with col3:
        transgender = st.number_input(
            "Transgender beneficiaries",
            min_value=0,
            value=0,
            step=1,
        )

    st.markdown("#### 2. Social-category distribution")

    col4, col5 = st.columns(2)

    with col4:
        sc = st.number_input(
            "SC beneficiaries",
            min_value=0,
            value=100,
            step=10,
        )

        st_count = st.number_input(
            "ST beneficiaries",
            min_value=0,
            value=50,
            step=10,
        )

    with col5:
        general = st.number_input(
            "General-category beneficiaries",
            min_value=0,
            value=700,
            step=10,
        )

        obc = st.number_input(
            "OBC beneficiaries",
            min_value=0,
            value=150,
            step=10,
        )

    st.markdown("#### 3. Coverage information")

    col6, col7 = st.columns(2)

    with col6:
        aadhaar = st.number_input(
            "Beneficiaries with Aadhaar",
            min_value=0,
            value=900,
            step=10,
        )

    with col7:
        mobile = st.number_input(
            "Beneficiaries with mobile numbers",
            min_value=0,
            value=700,
            step=10,
        )

    submitted = st.form_submit_button(
        "Predict Scheme Category",
        type="primary",
        use_container_width=True,
    )


# --------------------------------------------------
# Validate inputs and predict
# --------------------------------------------------

if submitted:

    errors = []

    gender_total = male + female + transgender
    social_total = sc + st_count + general + obc

    if gender_total > total:
        errors.append(
            "Male, female, and transgender counts cannot "
            "sum to more than total beneficiaries."
        )

    if social_total > total:
        errors.append(
            "Social-category counts cannot sum to more "
            "than total beneficiaries."
        )

    if aadhaar > total:
        errors.append(
            "Aadhaar count cannot exceed total beneficiaries."
        )

    if mobile > total:
        errors.append(
            "Mobile-number count cannot exceed total beneficiaries."
        )

    if errors:
        for error in errors:
            st.error(error)

    else:
        try:
            model = load_model()

            denominator = float(total)

            # Feature names and order must match training.
            features = pd.DataFrame(
                [
                    {
                        "male_share": male / denominator,
                        "female_share": female / denominator,
                        "transgender_share": transgender / denominator,
                        "sc_share": sc / denominator,
                        "st_share": st_count / denominator,
                        "general_share": general / denominator,
                        "obc_share": obc / denominator,
                        "aadhaar_coverage": aadhaar / denominator,
                        "mobile_coverage": mobile / denominator,
                    }
                ]
            )

            prediction = model.predict(features)[0]

            st.divider()
            st.subheader("Prediction Result")

            scheme_name = SCHEME_NAMES.get(
                str(prediction),
                str(prediction),
            )

            st.success(f"Predicted category: {prediction}")
            st.markdown(f"**Scheme:** {scheme_name}")

            # Display probabilities only if supported by the model.
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(features)[0]
                classes = model.classes_

                probability_df = pd.DataFrame(
                    {
                        "Scheme code": classes,
                        "Scheme name": [
                            SCHEME_NAMES.get(str(c), str(c))
                            for c in classes
                        ],
                        "Model score (%)": np.round(
                            probabilities * 100, 2
                        ),
                    }
                ).sort_values(
                    "Model score (%)",
                    ascending=False,
                )

                st.markdown("#### Model output scores")
                st.dataframe(
                    probability_df,
                    hide_index=True,
                    use_container_width=True,
                )

                st.caption(
                    "These scores are model-estimated class probabilities, "
                    "not verified eligibility probabilities or official "
                    "government decisions."
                )

            with st.expander("View the features supplied to the model"):
                st.dataframe(
                    features,
                    hide_index=True,
                    use_container_width=True,
                )

        except FileNotFoundError as exc:
            st.error(str(exc))
            st.info(
                "Ensure the trained model exists in the models folder."
            )

        except Exception as exc:
            st.error(
                "Prediction failed. Check the model file and feature "
                "names, then review the error below."
            )
            st.code(str(exc))


st.divider()

st.caption(
    "Project: NSAP Scheme Prediction using Machine Learning and IBM Cloud Lite"
)
