"""
Customer Churn Predictor — Premium Client Demo
Run with:  streamlit run app.py

Adds on top of the base model:
  - Per-customer SHAP explanation ("why is THIS customer at risk")
  - Revenue-at-risk framing in dollars (what churn actually costs)
  - Batch scoring with prioritized action list + total $ exposure
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Churn Predictor | Revenue Risk Dashboard",
                    page_icon="📉", layout="wide")

# ------------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("churn_model_pipeline.pkl")
    pre = model.named_steps["preprocess"]
    rf = model.named_steps["model"]
    explainer = shap.TreeExplainer(rf)

    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges",
                         "num_services", "avg_monthly_spend"]
    cat_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                     'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                     'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                     'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group']
    ohe = pre.named_transformers_["cat"]
    cat_names = ohe.get_feature_names_out(cat_features)
    feature_names = numeric_features + list(cat_names)

    # Map each one-hot encoded column back to its original categorical column,
    # so SHAP contributions can be grouped correctly (avoids misleading labels
    # like "Contract: Two year" showing up for a Month-to-month customer).
    feature_to_original = {f: f for f in numeric_features}
    for f in cat_names:
        for cf in cat_features:
            if f.startswith(cf + "_"):
                feature_to_original[f] = cf
                break

    return model, pre, rf, explainer, feature_names, cat_features, feature_to_original

model, preprocessor, rf_model, explainer, feature_names, cat_features, feature_to_original = load_model()

# ------------------------------------------------------------------
# FEATURE ENGINEERING (mirrors training exactly)
# ------------------------------------------------------------------
service_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                 "TechSupport", "StreamingTV", "StreamingMovies"]

def engineer_features(df):
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=[0, 12, 24, 48, 60, 72],
        labels=["0-1yr", "1-2yr", "2-4yr", "4-5yr", "5-6yr"], include_lowest=True
    )
    df["num_services"] = (df[service_cols] == "Yes").sum(axis=1)
    df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].replace(0, 1)
    return df

def explain_row(row_engineered):
    """Return top SHAP contributors for a single-row dataframe (already engineered),
    grouped by original column and paired with the customer's actual value —
    avoids showing a misleading one-hot column name for a category the
    customer doesn't actually have.

    NOTE: shap.TreeExplainer returns a different array shape depending on the
    model type. For RandomForestClassifier (binary classification) it returns
    shape (n_samples, n_features, 2) — one SHAP value per class per feature.
    We need the slice for class 1 (churn = Yes). XGBoost's binary classifier
    instead returns (n_samples, n_features) directly. Handling both shapes
    here makes this safe regardless of which model is loaded.
    """
    Xt = preprocessor.transform(row_engineered)
    sv = np.array(explainer.shap_values(Xt))

    if sv.ndim == 3:
        sv_row = sv[0, :, 1]   # sample 0, all features, positive (churn) class
    elif sv.ndim == 2:
        sv_row = sv[0]         # sample 0, all features
    else:
        raise ValueError(f"Unexpected SHAP output shape: {sv.shape}")

    df_sv = pd.DataFrame({"encoded_feature": feature_names, "shap": sv_row})
    df_sv["original_col"] = df_sv["encoded_feature"].map(feature_to_original)
    grouped = df_sv.groupby("original_col", as_index=False)["shap"].sum()
    grouped["abs_shap"] = grouped["shap"].abs()
    grouped = grouped.sort_values("abs_shap", ascending=False).head(6)
    grouped["actual_value"] = grouped["original_col"].apply(
        lambda c: str(row_engineered.iloc[0][c]))
    grouped["label"] = grouped["original_col"] + ": " + grouped["actual_value"]
    grouped["direction"] = np.where(grouped["shap"] > 0, "↑ increases risk", "↓ decreases risk")
    return grouped

# ------------------------------------------------------------------
# HEADER
# ------------------------------------------------------------------
st.title("📉 Customer Churn & Revenue Risk Dashboard")
st.caption("Random Forest model · ROC-AUC 0.843 · catches 76% of customers who actually churn "
           "· every prediction comes with a plain-English explanation")

tab1, tab2 = st.tabs(["🔍 Single Customer Lookup", "📁 Full Customer Base (Batch)"])

# ==================== TAB 1: SINGLE CUSTOMER ====================
with tab1:
    st.subheader("Enter customer details")
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

    with col2:
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])

    with col3:
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)

    total_charges = monthly_charges * max(tenure, 1)

    if st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True):
        row = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner, "Dependents": dependents, "tenure": tenure,
            "PhoneService": phone_service, "MultipleLines": multiple_lines,
            "InternetService": internet_service, "OnlineSecurity": online_security,
            "OnlineBackup": online_backup, "DeviceProtection": device_protection,
            "TechSupport": tech_support, "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies, "Contract": contract,
            "PaperlessBilling": paperless, "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
        }])
        row_eng = engineer_features(row)
        proba = model.predict_proba(row_eng)[0, 1]
        pred = model.predict(row_eng)[0]
        annual_value = monthly_charges * 12

        st.divider()
        c1, c2, c3 = st.columns([1, 1, 2])

        with c1:
            st.metric("Churn Probability", f"{proba:.0%}")
            if proba >= 0.6:
                st.error("🔴 High Risk")
            elif proba >= 0.3:
                st.warning("🟡 Medium Risk")
            else:
                st.success("🟢 Low Risk")

        with c2:
            st.metric("Customer's Annual Value", f"${annual_value:,.0f}")
            expected_loss = annual_value * proba
            st.metric("Revenue at Risk", f"${expected_loss:,.0f}",
                      help="Annual value × churn probability — the expected revenue loss if nothing is done.")

        with c3:
            st.write("**Why this prediction — top factors:**")
            explanation = explain_row(row_eng)
            for _, r in explanation.iterrows():
                bar_color = "#E74C3C" if r["shap"] > 0 else "#4CAF50"
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;padding:2px 0;'>"
                    f"<span>{r['label']}</span>"
                    f"<span style='color:{bar_color};font-weight:600;'>{r['direction']}</span>"
                    f"</div>", unsafe_allow_html=True
                )

        st.divider()
        if proba >= 0.6:
            st.error(f"**Recommended action:** Priority retention outreach — offer a contract "
                      f"upgrade incentive or loyalty discount. Losing this customer costs an "
                      f"estimated **${annual_value:,.0f}/year**.")
        elif proba >= 0.3:
            st.warning("**Recommended action:** Add to a monitoring list; a light-touch "
                        "check-in or loyalty perk can reduce risk.")
        else:
            st.info("**Recommended action:** No action needed — this customer looks stable.")

# ==================== TAB 2: BATCH ====================
with tab2:
    st.subheader("Upload your full customer list")
    st.caption("CSV must have the same columns as the training data. "
               "customerID and Churn columns are optional and will be ignored.")
    file = st.file_uploader("Choose CSV file", type="csv")

    if file is not None:
        batch = pd.read_csv(file)
        id_col = batch["customerID"] if "customerID" in batch.columns else pd.Series(
            [f"Customer {i+1}" for i in range(len(batch))])
        batch_clean = batch.drop(columns=[c for c in ["customerID", "Churn"] if c in batch.columns])
        batch_eng = engineer_features(batch_clean)

        probs = model.predict_proba(batch_eng)[:, 1]
        preds = model.predict(batch_eng)
        annual_value = batch_eng["MonthlyCharges"] * 12
        revenue_at_risk = annual_value * probs

        out = pd.DataFrame({
            "Customer": id_col,
            "Churn_Probability": probs.round(3),
            "Risk_Tier": pd.cut(probs, bins=[-0.01, 0.3, 0.6, 1.0],
                                 labels=["Low", "Medium", "High"]),
            "Monthly_Charges": batch_eng["MonthlyCharges"].round(2),
            "Annual_Value": annual_value.round(0),
            "Revenue_at_Risk": revenue_at_risk.round(0),
        }).sort_values("Revenue_at_Risk", ascending=False)

        total_at_risk = revenue_at_risk.sum()
        n_high_risk = int((probs >= 0.6).sum())

        st.divider()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Customers Scored", f"{len(out):,}")
        k2.metric("High-Risk Customers", f"{n_high_risk:,}")
        k3.metric("Total Revenue at Risk", f"${total_at_risk:,.0f}")
        k4.metric("Avg. Churn Probability", f"{probs.mean():.0%}")

        st.divider()
        st.write("**Customers ranked by revenue at risk — prioritize retention here first:**")
        st.dataframe(
            out.style.background_gradient(subset=["Churn_Probability"], cmap="Reds"),
            use_container_width=True, height=420
        )

        csv_out = out.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download prioritized retention list", csv_out,
                            "churn_risk_ranked.csv", "text/csv", type="primary")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("Model at a glance")
    st.metric("ROC-AUC", "0.843")
    st.metric("Recall (catches churners)", "76%")
    st.metric("Accuracy", "76.3%")
    st.divider()
    st.header("Top overall churn drivers")
    try:
        feat_imp = pd.read_csv("feature_importance.csv").head(8)
        feat_imp["feature"] = feat_imp["feature"].apply(
            lambda f: f"{feature_to_original.get(f, f)}: {f.split(feature_to_original.get(f, f) + '_')[-1]}"
            if f in feature_to_original and feature_to_original[f] != f else f
        )
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.barh(feat_imp["feature"][::-1], feat_imp["importance"][::-1], color="#3b82f6")
        ax.set_xlabel("Importance")
        st.pyplot(fig)
    except FileNotFoundError:
        st.caption("Run the training script first to generate feature_importance.csv")
