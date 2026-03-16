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
MODEL_DIR = BASE_DIR / "models" 

st.sidebar.header("Model Selection")

model_choice = st.sidebar.selectbox("Choose Model",("Gradient Boosting",
                                    "Random Forest","Decision Tree")
                                    )
@st.cache_resource
def load_model(model_name):
    model_paths = {
        "Gradient Boosting": MODEL_DIR / "gb_pipeline.pkl",
        "Random Forest": MODEL_DIR / "rf-pipeline.pkl",
        "Decision Tree": MODEL_DIR / "decision_tree.pkl"
    }
    return joblib.load(model_paths[model_name])

model = load_model(model_choice)
st.info(f"Currently Using **{model_choice} Model** ")
st.markdown(
"""
# 📊 Bank Marketing Deposit Prediction
### Machine Learning Dashboard
:blue-background[Predict whether a customer will subscribe to a **term deposit** using an Gradient Boosting model.
]
"""
)

st.sidebar.header("Customer Information")

age = st.sidebar.slider("Age",18,90,40)

job = st.sidebar.selectbox(
    "Job",
    ['admin.','technician','services','management','retired','blue-collar','student','entrepreneur']
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
    if model_choice == "Gradient Boosting":
        gb_model = model.named_steps["gb_trf"]
    elif model_choice == "Random Forest":
        gb_model =model.named_steps["rf_trf"]
    else:
        gb_model =model.named_steps["dt_trf"]
    encoding = model.named_steps["trf1"]

    feature_names = [f.split("__")[-1] for f in encoding.get_feature_names_out()]
    importances = gb_model.feature_importances_

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

    st.subheader("SHAP Feature Impact for 500 customers")

    encoding = model.named_steps["trf1"]
    if model_choice == "Gradient Boosting":
        xgb_model = model.named_steps["gb_trf"]
    elif model_choice == "Random Forest":
        xgb_model =model.named_steps["rf_trf"]
    else:
        xgb_model =model.named_steps["dt_trf"]

    X_transformed = encoding.transform(input_data)

    explainer = shap.TreeExplainer(xgb_model)
    X_sample = encoding.transform(X.sample(500))

    shap_values = explainer.shap_values(X_sample)

  # Handle binary classification output
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    fig = plt.figure()

    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=encoding.get_feature_names_out(),
        show=False
    )

    st.pyplot(fig)

st.markdown("---")

st.markdown(
"""
Built with **Streamlit + Gradient Boosting + SHAP**

"""
)