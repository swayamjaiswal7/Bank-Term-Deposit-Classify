import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import shap

# Load model
st.set_page_config(
    page_title="Bank Marketing ML Dashboard",
    page_icon="💹",
    layout="wide"
)
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "pipeline_xgb.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()
st.markdown(
"""
# 📊 Bank Marketing Deposit Prediction
### Machine Learning Dashboard
:blue-background[Predict whether a customer will subscribe to a **term deposit** using an XGBoost model.
]
"""
)

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

tab1, tab2, tab3, tab4 = st.tabs([
    "Prediction",
    "Feature Importance",
    "Precision Recall Curve",
    "SHAP Explanation"
])

# --------------------
# Prediction Tab
# --------------------

with tab1:

    if st.button("Predict Subscription Probability"):

        prob = model.predict_proba(input_data)[0][1]

        st.metric("Subscription Probability", f"{prob:.2f}")

        if prob > 0.65:
            st.success("High probability customer")
        elif prob > 0.40:
            st.warning("Moderate probability")
        else:
            st.error("Low probability")

# --------------------
# Feature Importance
# --------------------

with tab2:

    st.subheader("Feature Importance")

    xgb_model = model.named_steps["xgb"]
    encoding = model.named_steps["enc"]

    feature_names = encoding.get_feature_names_out()

    importances = xgb_model.feature_importances_

    feat_df = pd.DataFrame({
        "feature":feature_names,
        "importance":importances
    }).sort_values("importance",ascending=False)

    fig, ax = plt.subplots()
    ax.barh(feat_df["feature"][:15], feat_df["importance"][:15])
    ax.invert_yaxis()

    st.pyplot(fig)

# --------------------
# Precision Recall Curve
# --------------------
X=pd.read_csv("./data/bank-marketing-cleaned.csv"
                   ,usecols=['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
       'loan', 'contact', 'campaign', 'pdays', 'previous'])
y = pd.read_csv("./data/outcome.csv",usecols=['y']).squeeze()

with tab3:

    st.subheader("Precision Recall Curve")

    X_sample = X.sample(2000, random_state=42)
    y_sample = y.loc[X_sample.index]
    y_sample = y_sample.map({'no':0,'yes':1})
    y_prob = model.predict_proba(X_sample)[:,1]

    precision, recall, thresholds = precision_recall_curve(y_sample, y_prob)

    fig, ax = plt.subplots()

    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision Recall Curve")

    st.pyplot(fig)

# --------------------
# SHAP Explanation
# --------------------

with tab4:

    st.subheader("SHAP Feature Impact")

    encoding = model.named_steps["enc"]
    xgb_model = model.named_steps["xgb"]

    X_transformed = encoding.transform(input_data)

    explainer = shap.TreeExplainer(xgb_model)

    shap_values = explainer.shap_values(X_transformed)

    fig = plt.figure()

    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=encoding.get_feature_names_out(),
        show=False
    )

    st.pyplot(fig)

st.markdown("---")

st.markdown(
"""
Built with **Streamlit + XGBoost + SHAP**

"""
)