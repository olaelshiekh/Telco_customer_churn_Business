import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "churn_pipeline.joblib"
META_PATH = BASE_DIR / "models" / "metadata.json"

INTERNET_ADDONS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
DEFAULTS = {
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check",
}
LABELS = {
    "SeniorCitizen": "Senior citizen",
    "PhoneService": "Phone service",
    "MultipleLines": "Multiple lines",
    "InternetService": "Internet service",
    "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection",
    "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies",
    "PaperlessBilling": "Paperless billing",
    "PaymentMethod": "Payment method",
}


@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    return model, meta


def clean(value) -> str:
    """Category values from OpenML keep literal quotes, e.g. "'Two year'"."""
    return str(value).strip("'")


def display_name(col: str) -> str:
    return LABELS.get(col, col)


def display_option(col: str, value) -> str:
    if col == "SeniorCitizen":
        return "Yes" if int(value) == 1 else "No"
    return clean(value)


def pick(col: str, meta: dict, container, forced: str = None):
    """Selectbox over a column's known categories; returns the raw category value."""
    raw_options = meta["label_mappings"][col]
    if forced is not None:
        raw = next(v for v in raw_options if clean(v) == forced)
        container.selectbox(
            display_name(col),
            [forced],
            disabled=True,
            key=f"{col}_forced",
            help="Set automatically from the related service choice.",
        )
        return raw
    default = DEFAULTS.get(col)
    index = next((i for i, v in enumerate(raw_options) if clean(v) == default), 0)
    return container.selectbox(
        display_name(col),
        raw_options,
        index=index,
        format_func=lambda v: display_option(col, v),
        key=col,
    )


def encode(raw_values: dict, meta: dict) -> pd.DataFrame:
    """Reproduce the notebook's LabelEncoder step: category -> index in the sorted list."""
    row = {}
    for col in meta["feature_columns"]:
        if col in meta["label_mappings"]:
            row[col] = meta["label_mappings"][col].index(raw_values[col])
        else:
            row[col] = raw_values[col]
    return pd.DataFrame([row])[meta["feature_columns"]]


def confidence_band(conf: float) -> str:
    if conf >= 0.80:
        return "High"
    if conf >= 0.65:
        return "Moderate"
    return "Low"


def main():
    st.set_page_config(page_title="Telco Churn Predictor", page_icon="📞", layout="wide")
    st.title("📞 Telco Customer Churn Predictor")
    st.caption(
        "Enter the customer's details to estimate whether they are likely to leave the company."
    )

    try:
        model, meta = load_artifacts()
    except FileNotFoundError:
        st.error(
            "Model files not found. Run the notebook first so that "
            "`models/churn_pipeline.joblib` and `models/metadata.json` exist."
        )
        st.stop()

    values = {}

    st.subheader("1. Customer profile")
    c1, c2, c3, c4 = st.columns(4)
    values["gender"] = pick("gender", meta, c1)
    values["SeniorCitizen"] = pick("SeniorCitizen", meta, c2)
    values["Partner"] = pick("Partner", meta, c3)
    values["Dependents"] = pick("Dependents", meta, c4)

    st.subheader("2. Services")
    c1, c2 = st.columns(2)
    values["PhoneService"] = pick("PhoneService", meta, c1)
    no_phone = clean(values["PhoneService"]) == "No"
    values["MultipleLines"] = pick(
        "MultipleLines", meta, c2, forced="No phone service" if no_phone else None
    )

    values["InternetService"] = pick("InternetService", meta, c1)
    no_internet = clean(values["InternetService"]) == "No"

    cols = st.columns(3)
    for i, col in enumerate(INTERNET_ADDONS):
        values[col] = pick(
            col, meta, cols[i % 3], forced="No internet service" if no_internet else None
        )

    st.subheader("3. Contract & billing")
    c1, c2, c3 = st.columns(3)
    values["Contract"] = pick("Contract", meta, c1)
    values["PaperlessBilling"] = pick("PaperlessBilling", meta, c2)
    values["PaymentMethod"] = pick("PaymentMethod", meta, c3)

    c1, c2, c3 = st.columns(3)
    values["tenure"] = c1.number_input(
        "Tenure (months)", min_value=0, max_value=100, value=12, step=1
    )
    values["MonthlyCharges"] = c2.number_input(
        "Monthly charges ($)", min_value=0.0, max_value=500.0, value=70.0, step=1.0
    )
    values["TotalCharges"] = c3.number_input(
        "Total charges ($)",
        min_value=0.0,
        max_value=20000.0,
        value=840.0,
        step=10.0,
        help="Total amount billed so far (roughly tenure x monthly charges).",
    )

    if st.button("Predict churn", type="primary", width="stretch"):
        X = encode(values, meta)
        proba = model.predict_proba(X)[0]
        churn_p = float(proba[list(model.classes_).index(1)])
        will_leave = churn_p >= 0.5
        confidence = max(churn_p, 1 - churn_p)
        band = confidence_band(confidence)

        st.divider()
        if will_leave:
            st.error(
                f"⚠️ **This customer is likely to LEAVE.** "
                f"Model confidence: **{confidence:.1%}** ({band})."
            )
        else:
            st.success(
                f"✅ **This customer is likely to STAY.** "
                f"Model confidence: **{confidence:.1%}** ({band})."
            )

        m1, m2, m3 = st.columns(3)
        m1.metric("Churn probability", f"{churn_p:.1%}")
        m2.metric("Retention probability", f"{1 - churn_p:.1%}")
        m3.metric("Confidence", f"{confidence:.1%}", band, delta_color="off")
        st.progress(churn_p, text=f"Churn risk: {churn_p:.1%}")

        if band == "Low":
            st.info(
                "The prediction is close to a coin flip. Treat it as a weak signal "
                "and rely on additional context before acting."
            )
        if will_leave:
            st.markdown(
                "**Suggested action:** consider a retention offer, e.g. a longer-term "
                "contract discount or a service bundle."
            )

        with st.expander("Input sent to the model"):
            st.dataframe(
                pd.DataFrame([{k: clean(v) for k, v in values.items()}]),
                width="stretch",
            )


main()
