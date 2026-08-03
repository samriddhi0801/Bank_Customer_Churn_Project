# ============================================================
# PROFESSIONAL BANK CUSTOMER CHURN DASHBOARD
# PART 1A
# Copy everything into app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

import plotly.express as px
import plotly.graph_objects as go

from datetime import datetime

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# PROFESSIONAL CSS
# ---------------------------------------------------------

st.markdown("""

<style>

#MainMenu{visibility:hidden;}
footer{visibility:hidden;}


.block-container{
padding-top:1rem;
padding-bottom:2rem;
padding-left:2rem;
padding-right:2rem;
}

[data-testid="stSidebar"]{

background:linear-gradient(180deg,#001F3F,#0A4D68);

}

[data-testid="stSidebar"] *{

color:white;

}

.main-title{

font-size:42px;

font-weight:800;

color:#0B3C5D;

}

.sub-title{

font-size:18px;

color:gray;

margin-bottom:15px;

}

.metric-card{

background:white;

padding:18px;

border-radius:15px;

box-shadow:0 4px 12px rgba(0,0,0,.15);

text-align:center;

}

.section{

background:#F8FAFC;

padding:20px;

border-radius:15px;

margin-bottom:15px;

}

</style>

""",unsafe_allow_html=True)

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
BASE_DIR,
"..",
"models",
"best_churn_model.pkl"
)

SCALER_PATH = os.path.join(
BASE_DIR,
"..",
"models",
"scaler.pkl"
)

DATA_PATH = os.path.join(
BASE_DIR,
"..",
"data",
"European_Bank.csv"
)

# ---------------------------------------------------------
# LOAD FILES
# ---------------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

model = load_model()
scaler = load_scaler()
df = load_data()

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown(
"""
<div class='main-title'>
🏦 Predictive Modeling and Risk Scoring
</div>
""",
unsafe_allow_html=True
)

st.markdown(
"""
<div class='sub-title'>
European Bank Customer Churn Prediction System
</div>
""",
unsafe_allow_html=True
)

st.markdown("---")

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.image(
"https://img.icons8.com/fluency/240/bank-building.png",
width=120
)

st.sidebar.title("Navigation")

page = st.sidebar.radio(

"Select Module",

[
"🏠 Home",
"📊 Dashboard",
"🤖 Prediction",
"📈 Feature Importance",
"📉 Model Performance",
"🔄 What-if Analysis",
"📄 Executive Summary",
"ℹ️ About"
]

)

st.sidebar.markdown("---")

st.sidebar.success("✅ Model Loaded")

st.sidebar.info("Gradient Boosting Classifier")

st.sidebar.write("")

st.sidebar.metric(
"Dataset",
f"{len(df):,} Customers"
)

st.sidebar.metric(
"Features",
str(df.shape[1])
)

st.sidebar.metric(
"Date",
datetime.now().strftime("%d-%m-%Y")
)

st.sidebar.markdown("---")

st.sidebar.caption(
"Developed using Streamlit"
)
# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    st.markdown("## 🏦 Bank Customer Churn Intelligence Dashboard")

    st.markdown("""
This dashboard predicts customer churn risk using a trained **Gradient Boosting Machine Learning Model**.
It helps banks identify high-risk customers, improve retention strategies and make data-driven business decisions.
""")

    st.write("")

    total_customers = len(df)
    churn_customers = int(df["Exited"].sum())
    retained_customers = total_customers - churn_customers
    churn_rate = round((churn_customers/total_customers)*100,2)
    avg_credit = round(df["CreditScore"].mean(),0)
    avg_balance = round(df["Balance"].mean(),2)
    avg_salary = round(df["EstimatedSalary"].mean(),2)

    c1,c2,c3,c4 = st.columns(4)

    with c1:
        st.metric(
            "👥 Total Customers",
            f"{total_customers:,}"
        )

    with c2:
        st.metric(
            "❌ Churn Customers",
            churn_customers
        )

    with c3:
        st.metric(
            "✅ Retained Customers",
            retained_customers
        )

    with c4:
        st.metric(
            "📊 Churn Rate",
            f"{churn_rate}%"
        )

    st.write("")

    c5,c6,c7 = st.columns(3)

    with c5:
        st.metric(
            "💳 Avg Credit Score",
            avg_credit
        )

    with c6:
        st.metric(
            "🏦 Avg Balance",
            f"${avg_balance:,.0f}"
        )

    with c7:
        st.metric(
            "💰 Avg Salary",
            f"${avg_salary:,.0f}"
        )

    st.markdown("---")

    left,right = st.columns(2)

    with left:

        pie_df = pd.DataFrame({

            "Status":[
                "Retained",
                "Churned"
            ],

            "Customers":[
                retained_customers,
                churn_customers
            ]

        })

        fig = px.pie(

            pie_df,

            names="Status",

            values="Customers",

            hole=.60,

            color_discrete_sequence=[
                "#2ECC71",
                "#E74C3C"
            ]

        )

        fig.update_layout(
            title="Customer Distribution",
            height=450
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        geo = df.groupby(
            "Geography"
        )["Exited"].mean().reset_index()

        geo["Exited"] *=100

        fig = px.bar(

            geo,

            x="Geography",

            y="Exited",

            text_auto=".1f",

            color="Exited",

            title="Churn Rate by Geography"

        )

        fig.update_layout(
            height=450
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    st.subheader("📌 Business Insights")

    col1,col2,col3 = st.columns(3)

    with col1:

        st.success("""

### Customer Analytics

✔ Customer Segmentation

✔ Banking Behaviour

✔ Churn Monitoring

✔ Customer Insights

""")

    with col2:

        st.info("""

### Machine Learning

✔ Gradient Boosting

✔ Risk Prediction

✔ Probability Score

✔ Explainable AI

""")

    with col3:

        st.warning("""

### Business Value

✔ Customer Retention

✔ Revenue Protection

✔ Risk Management

✔ Decision Support

""")

    st.markdown("---")

    st.subheader("📈 Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )
    # ============================================================
# ANALYTICS DASHBOARD
# ============================================================

if page == "📊 Dashboard":

    st.title("📊 Customer Analytics Dashboard")

    st.markdown("Analyze customer demographics, financial behaviour and churn trends.")

    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    with c1:
        geography = st.selectbox(
            "🌍 Geography",
            ["All"] + sorted(df["Geography"].unique().tolist())
        )

    with c2:
        gender = st.selectbox(
            "👤 Gender",
            ["All"] + sorted(df["Gender"].unique().tolist())
        )

    with c3:
        active = st.selectbox(
            "⭐ Active Member",
            ["All", 0, 1]
        )

    data = df.copy()

    if geography != "All":
        data = data[data["Geography"] == geography]

    if gender != "All":
        data = data[data["Gender"] == gender]

    if active != "All":
        data = data[data["IsActiveMember"] == active]

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        fig = px.histogram(
            data,
            x="Age",
            nbins=25,
            color="Exited",
            title="Age Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        fig = px.box(
            data,
            x="Exited",
            y="Balance",
            color="Exited",
            title="Balance vs Churn"
        )

        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:

        credit = (
            data.groupby("Exited")["CreditScore"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            credit,
            x="Exited",
            y="CreditScore",
            color="Exited",
            title="Average Credit Score"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col4:

        geo = (
            data.groupby("Geography")["Exited"]
            .mean()
            .reset_index()
        )

        geo["Exited"] *= 100

        fig = px.bar(
            geo,
            x="Geography",
            y="Exited",
            color="Exited",
            text_auto=".1f",
            title="Churn Rate by Geography (%)"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    col5, col6 = st.columns(2)

    with col5:

        active_df = (
            data.groupby("IsActiveMember")["Exited"]
            .mean()
            .reset_index()
        )

        active_df["Exited"] *= 100

        active_df["IsActiveMember"] = active_df[
            "IsActiveMember"
        ].replace({
            0: "Inactive",
            1: "Active"
        })

        fig = px.bar(
            active_df,
            x="IsActiveMember",
            y="Exited",
            color="Exited",
            title="Active Members vs Churn"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col6:

        fig = px.scatter(
            data,
            x="EstimatedSalary",
            y="Balance",
            color="Exited",
            size="Age",
            hover_data=["Geography", "Gender"],
            title="Balance vs Salary"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Correlation Matrix")

    corr = data.select_dtypes(include=np.number).corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        aspect="auto"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Filtered Dataset")

    st.dataframe(
        data,
        use_container_width=True
    )
    # ============================================================
# MODEL PERFORMANCE
# ============================================================

if page == "📉 Model Performance":

    st.title("📉 Machine Learning Model Performance")

    st.markdown(
        "Performance of the trained **Gradient Boosting Classifier** on the test dataset."
    )

    st.markdown("---")

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric(
            "Accuracy",
            "86.95%"
        )

    with c2:
        st.metric(
            "Precision",
            "78.97%"
        )

    with c3:
        st.metric(
            "Recall",
            "48.89%"
        )

    with c4:
        st.metric(
            "F1 Score",
            "60.39%"
        )

    with c5:
        st.metric(
            "ROC-AUC",
            "86.93%"
        )

    st.markdown("---")

    performance = pd.DataFrame({

        "Metric":[
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC"
        ],

        "Score":[
            86.95,
            78.97,
            48.89,
            60.39,
            86.93
        ]

    })

    fig = px.bar(

        performance,

        x="Metric",

        y="Score",

        color="Score",

        text="Score",

        title="Model Performance Metrics"

    )

    fig.update_traces(texttemplate="%{text:.2f}%")

    fig.update_layout(height=500)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    st.success("""
### Model Interpretation

✔ Excellent overall accuracy (86.95%)

✔ Strong ROC-AUC indicates good discrimination between churn and non-churn customers.

✔ High precision reduces unnecessary retention campaigns.

✔ Recall can be improved in future versions using class balancing or hyperparameter tuning.
""")
    # ============================================================
# WHAT-IF ANALYSIS
# ============================================================

if page == "🔄 What-if Analysis":

    st.title("🔄 What-if Scenario Simulator")

    st.write(
        "Modify customer information and observe how churn risk changes."
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        age = st.slider(
            "Age",
            18,
            90,
            35
        )

        credit = st.slider(
            "Credit Score",
            300,
            900,
            650
        )

        balance = st.number_input(
            "Balance",
            0.0,
            300000.0,
            60000.0,
            step=1000.0
        )

        salary = st.number_input(
            "Estimated Salary",
            1000.0,
            300000.0,
            70000.0,
            step=1000.0
        )

    with col2:

        tenure = st.slider(
            "Tenure",
            0,
            10,
            5
        )

        products = st.slider(
            "Number of Products",
            1,
            4,
            2
        )

        active = st.selectbox(
            "Active Member",
            [0,1]
        )

        card = st.selectbox(
            "Credit Card",
            [0,1]
        )

        geography = st.selectbox(
            "Geography",
            ["France","Germany","Spain"]
        )

        gender = st.selectbox(
            "Gender",
            ["Female","Male"]
        )

    germany = 1 if geography=="Germany" else 0
    spain = 1 if geography=="Spain" else 0
    male = 1 if gender=="Male" else 0

    balance_salary = balance/salary if salary>0 else 0

    product_density = products

    engagement = active*products

    age_tenure = age*tenure

    sample = pd.DataFrame({

        "Year":[2025],
        "CreditScore":[credit],
        "Age":[age],
        "Tenure":[tenure],
        "Balance":[balance],
        "NumOfProducts":[products],
        "HasCrCard":[card],
        "IsActiveMember":[active],
        "EstimatedSalary":[salary],
        "Balance_to_Salary_Ratio":[balance_salary],
        "Product_Density":[product_density],
        "Engagement_Product_Interaction":[engagement],
        "Age_Tenure_Interaction":[age_tenure],
        "Geography_Germany":[germany],
        "Geography_Spain":[spain],
        "Gender_Male":[male]

    })

    probability = model.predict_proba(sample)[0][1]

    st.markdown("---")

    st.metric(
        "Predicted Churn Probability",
        f"{probability*100:.2f}%"
    )

    gauge = go.Figure(go.Indicator(

        mode="gauge+number",

        value=probability*100,

        title={"text":"Risk Score"},

        gauge={

            "axis":{"range":[0,100]},

            "bar":{"color":"darkblue"},

            "steps":[

                {"range":[0,30],"color":"lightgreen"},

                {"range":[30,70],"color":"gold"},

                {"range":[70,100],"color":"red"}

            ]

        }

    ))

    gauge.update_layout(height=420)

    st.plotly_chart(
        gauge,
        use_container_width=True
    )

    if probability < 0.30:

        st.success("🟢 Low Risk Customer")

    elif probability < 0.70:

        st.warning("🟡 Medium Risk Customer")

    else:

        st.error("🔴 High Risk Customer")
        # ============================================================
# EXECUTIVE SUMMARY
# ============================================================

if page == "📄 Executive Summary":

    st.title("📄 Executive Summary")

    st.markdown("""
## Project Overview

This project predicts customer churn using Machine Learning to help banks
identify customers likely to leave and take proactive retention actions.

---

## Business Problem

Banks lose revenue when valuable customers leave.

This dashboard helps identify:

- High Risk Customers
- Customer Behaviour
- Churn Drivers
- Retention Opportunities

---

## Machine Learning Model

**Algorithm**

✅ Gradient Boosting Classifier

---

## Model Performance

| Metric | Score |
|--------|--------|
| Accuracy | 86.95% |
| Precision | 78.97% |
| Recall | 48.89% |
| F1 Score | 60.39% |
| ROC-AUC | 86.93% |

---

## Major Churn Drivers

• Age

• Number of Products

• Balance

• Customer Engagement

• Active Membership

• Geography

---

## Business Recommendations

✅ Increase customer engagement.

✅ Improve loyalty programs.

✅ Offer personalized banking products.

✅ Contact high-risk customers before churn.

✅ Encourage active membership.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Plotly
- Streamlit
- Joblib

""")

# ============================================================
# ABOUT PROJECT
# ============================================================

if page == "ℹ️ About":

    st.title("ℹ️ About Project")

    st.markdown("""

## Predictive Modeling and Risk Scoring for Bank Customer Churn

This project was developed to predict customer churn using
Machine Learning and provide business-friendly insights
through an interactive Streamlit dashboard.

### Features

✅ Interactive Dashboard

✅ Customer Churn Prediction

✅ Risk Probability

✅ Feature Importance

✅ Model Performance

✅ What-if Analysis

✅ Executive Summary

---

### Dataset

European Bank Customer Dataset

---

### Developed Using

Python

Pandas

NumPy

Scikit-Learn

Plotly

Streamlit

Joblib

""")

st.markdown("---")

st.markdown(
"""
<div style='text-align:center;color:gray;font-size:15px;'>

🏦 Predictive Modeling and Risk Scoring for Bank Customer Churn

Developed using Streamlit & Machine Learning

© 2026 All Rights Reserved

</div>
""",
unsafe_allow_html=True)
# ============================================================
# BUSINESS RECOMMENDATIONS
# ============================================================

st.markdown("---")

st.header("💼 Strategic Business Recommendations")

col1, col2 = st.columns(2)

with col1:
    st.success("""
### 🎯 Customer Retention

✔ Launch loyalty reward programs

✔ Contact high-risk customers proactively

✔ Improve digital banking experience

✔ Provide personalized offers

✔ Increase customer engagement
""")

with col2:
    st.info("""
### 📈 Business Growth

✔ Cross-sell banking products

✔ Improve customer satisfaction

✔ Increase active membership

✔ Reduce customer attrition

✔ Improve long-term profitability
""")

st.markdown("---")

st.header("📌 Project Conclusion")

st.write("""
This dashboard predicts customer churn using a Gradient Boosting Machine Learning model.
It enables banks to identify customers at high risk of leaving and supports proactive
retention strategies through predictive analytics, interactive visualizations,
risk scoring and scenario analysis.

The solution helps improve customer lifetime value, reduce churn and support
better business decision-making.
""")

st.markdown("---")

st.caption(
    "Developed for Unified Mentor | Predictive Modeling and Risk Scoring for Bank Customer Churn"
)
