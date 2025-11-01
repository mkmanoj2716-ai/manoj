import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# --------------------------
# PAGE CONFIG
# --------------------------
st.set_page_config(
    page_title="Optimizing ROI on Digital Marketing Campaigns",
    page_icon="✈️",
    layout="wide"
)

# --------------------------
# PROJECT HEADER SECTION
# --------------------------
st.title("📊 Optimizing ROI (Return on Investment) on Digital Marketing Campaigns")
st.subheader("Developed by: **Manoj** | MBA – Data Analytics")

st.markdown("""
---

### 📝 **Project Overview**

This project, **“Optimizing ROI (Return on Investment) on Digital Marketing Campaigns,”** focuses on helping airlines
make smarter, data-driven marketing decisions.  
Using **machine learning models** and **customer behavioral analytics**, the system identifies
which travelers are most likely to purchase flight tickets, helping optimize digital ad spend and maximize conversions.

Developed under **AeroReach Insights**, this project aligns technology and marketing intelligence
to improve how airlines engage with travelers online.

---

### 🎯 **Objectives**
- Identify **high-value travelers** likely to purchase tickets.  
- Analyze **engagement metrics** (likes, comments, views, time spent) that drive conversions.  
- Build **device-specific models** (mobile vs. desktop) for better targeting.  
- Evaluate **content engagement** and its impact on ticket purchases.  
- Deliver **data-driven insights** that increase ROI on digital marketing.

---

### ✅ **Project Requirements Satisfied**
| Requirement | How It’s Achieved |
|--------------|------------------|
| **Predict purchase likelihood** | Logistic Regression & Random Forest (SMOTE for imbalance) |
| **Optimize ROI** | Target high-probability users to reduce wasted spend |
| **Platform behavior analysis** | Separate modeling for Mobile & Desktop |
| **Measure content engagement** | Analyze likes, comments, and travel views vs. purchases |
| **Visualization & interpretation** | Streamlit dashboard with predictions and feature importance |

---

This solution enables airlines to **focus digital campaigns on customers most likely to convert**,  
reducing marketing costs and **maximizing ROI** through precision-driven targeting.

---
""")

# --------------------------
# LOAD MODEL
# --------------------------
@st.cache_resource
def load_model():
    model_path = "models/final_rf_smote.joblib"
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

if model is not None:
    st.success("✅ Model loaded successfully!")
else:
    st.stop()

# --------------------------
# USER INPUT FORM
# --------------------------
st.header("🔧 Input Traveler Profile")

col1, col2, col3 = st.columns(3)
with col1:
    Yearly_avg_view_on_travel_page = st.number_input("Yearly Avg Views on Travel Page", 0, 2000, 250)
    member_in_family = st.number_input("Family Members", 1, 10, 2)
    Daily_Avg_mins_spend_on_traveling_page = st.number_input("Daily Avg Minutes Spent on Travel Page", 0, 300, 20)
with col2:
    engagement_score = st.number_input("Engagement Score (likes/comments normalized)", -10.0, 10.0, 0.0)
    week_since_last_outstation_checkin = st.number_input("Weeks Since Last Outstation Check-in", 0, 52, 4)
    travelling_network_rating = st.number_input("Travel Network Rating (1–5)", 1, 5, 3)
with col3:
    montly_avg_comment_on_company_page = st.number_input("Monthly Avg Comments on Company Page", 0, 100, 5)
    device_type = st.selectbox("Preferred Device", ["mobile", "desktop", "unknown"])
    loc_type_simple = st.selectbox("Preferred Location Type", ["business", "medical", "other"])
    working_flag = st.selectbox("Working Flag", ["Yes", "No"])
    following_company_page = st.selectbox("Following Company Page", ["Yes", "No"])
    adult_flag = st.slider("Age Group (Numeric flag)", 18, 70, 36)

# Create dataframe for model
input_dict = {
    'Yearly_avg_view_on_travel_page': [Yearly_avg_view_on_travel_page],
    'member_in_family': [member_in_family],
    'Daily_Avg_mins_spend_on_traveling_page': [Daily_Avg_mins_spend_on_traveling_page],
    'engagement_score': [engagement_score],
    'week_since_last_outstation_checkin': [week_since_last_outstation_checkin],
    'travelling_network_rating': [travelling_network_rating],
    'montly_avg_comment_on_company_page': [montly_avg_comment_on_company_page],
    'device_type': [device_type],
    'loc_type_simple': [loc_type_simple],
    'working_flag': [working_flag],
    'following_company_page': [following_company_page],
    'adult_flag': [adult_flag]
}
input_df = pd.DataFrame(input_dict)

st.subheader("📋 Input Summary")
st.dataframe(input_df)

# --------------------------
# PREDICTION
# --------------------------
if st.button("🔮 Predict Conversion Probability"):
    try:
        proba = model.predict_proba(input_df)[:, 1][0]
        pred = model.predict(input_df)[0]

        st.success(f"**Predicted Conversion Probability: {proba:.2%}**")

        # Simple ROI estimation: assume marketing spend & ticket price
        avg_ticket_price = 300  # hypothetical
        avg_campaign_cost = 50
        expected_revenue = proba * avg_ticket_price
        expected_roi = ((expected_revenue - avg_campaign_cost) / avg_campaign_cost) * 100

        st.metric(label="💰 Estimated ROI if Targeted (%)", value=f"{expected_roi:.1f}%")

        if pred == 1:
            st.markdown("🎯 **High-potential traveler!** Recommend personalized ads or loyalty offers.")
        else:
            st.markdown("💤 **Low conversion likelihood.** Consider awareness or engagement campaigns first.")

    except Exception as e:
        st.error(f"Prediction error: {e}")

# --------------------------
# FEATURE IMPORTANCE
# --------------------------
st.header("📈 Feature Importance – Top ROI Drivers")
try:
    clf = model.named_steps['clf'] if 'clf' in model.named_steps else model
    if hasattr(clf, "feature_importances_"):
        pre = model.named_steps['pre']
        num_features = pre.transformers_[0][2]
        cat_encoder = pre.transformers_[1][1].named_steps['ohe']
        cat_cols = pre.transformers_[1][2]
        feat_names = list(num_features) + list(cat_encoder.get_feature_names_out(cat_cols))
        importances = pd.Series(clf.feature_importances_, index=feat_names).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(8,5))
        importances.tail(15).plot(kind='barh', ax=ax, color="#1f77b4")
        plt.title("Top 15 Most Influential Features (ROI Drivers)")
        st.pyplot(fig)
    else:
        st.info("Feature importances unavailable for this model.")
except Exception as e:
    st.warning(f"Could not extract feature importance: {e}")

# --------------------------
# FOOTER / CREDITS
# --------------------------
st.markdown("---")
st.caption("""
📘 **MBA Analytics Project – Optimizing ROI on Digital Marketing Campaigns**  
Developed by: **Manoj**  

""")
