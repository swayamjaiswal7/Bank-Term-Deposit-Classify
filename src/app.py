import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Load model

model = joblib.load(".\models\pipeline_xgb.pkl")

st.title("Bank Marketing Deposit Prediction")

st.write("""
This application predicts whether a customer is likely to subscribe
to a **term deposit** based on demographic and campaign data.
""")

st.sidebar.header("Customer Information")

age = st.sidebar.slider("Age",18,90,40)

job = st.sidebar.selectbox(
    "Job",
    ['admin','technician','services','management','retired','blue-collar','student','entrepreneur']
)

marital = st.sidebar.selectbox(
    "Marital Status",
    ['single','married','divorced']
)

education = st.sidebar.selectbox(
    "Education",
    ['primary','secondary','tertiary']
)

default = st.sidebar.selectbox(
    "Credit Default",
    ['no','yes']
)

balance = st.sidebar.number_input(
    "Account Balance",
    value=1000
)

housing = st.sidebar.selectbox(
    "Housing Loan",
    ['no','yes']
)

loan = st.sidebar.selectbox(
    "Personal Loan",
    ['no','yes']
)

contact = st.sidebar.selectbox(
    "Contact Type",
    ['cellular','telephone']
)

campaign = st.sidebar.slider("Number of Campaign Contacts",1,50,2)

pdays = st.sidebar.slider("Days Since Last Contact",-1,500,100)

previous = st.sidebar.slider("Previous Contacts",0,20,0)

# Create dataframe
input_data = pd.DataFrame({
    'age':[age],
    'job':[job],
    'marital':[marital],
    'education':[education],
    'default':[default],
    'balance':[balance],
    'housing':[housing],
    'loan':[loan],
    'contact':[contact],
    'campaign':[campaign],
    'pdays':[pdays],
    'previous':[previous]
})

if st.button("Predict Subscription Probability"):

    prob = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    st.write(f"Subscription Probability: **{prob:.2f}**")

    if prob > 0.65:
        st.success("High probability customer — target for marketing campaign")
    elif prob > 0.40:
        st.warning("Moderate probability customer")
    else:
        st.error("Low probability customer — avoid costly outreach")