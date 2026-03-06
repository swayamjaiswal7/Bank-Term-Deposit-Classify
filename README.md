# Bank Marketing Campaign Analytics & Deposit Prediction (Dashboard + ML Classification)

## Project Overview
This project combines **Marketing Campaign Analytics** with **Machine Learning Classification** to solve a real-world business problem:

> **Predict whether a customer will subscribe to a term deposit** based on demographic, financial, and campaign interaction data.

The project is built in two major layers:

 **Interactive Marketing Dashboard** (Campaign Performance Monitoring)  
**Machine Learning Classification Models** (Deposit Subscription Prediction)

## 🎯 Project Objectives

### 1️ Business Objectives
- Understand customer behavior during marketing campaigns.
- Identify which campaign channels generate maximum conversions.
- Analyze customer engagement through website visits, email interactions, and time spent.
- Segment customers by campaign type (Awareness, Consideration, Conversion, Retention).
- Improve marketing ROI by reducing unnecessary outreach.

### 2️ Machine Learning Objectives
- Build a predictive model that classifies customers into:
  - **Deposit Subscriber (Yes)**
  - **Non-Subscriber (No)**
- Compare multiple ML models and evaluate performance.
- Select the best model using classification metrics (not just accuracy).
- Identify which model best balances fraud-like class imbalance behavior (minority positive class).

---

#  Dataset Description

The dataset used is a cleaned version of the **Bank Marketing dataset**, containing 45,211 customer records.

### Features Used:
| Feature | Description |
|--------|-------------|
| age | Customer age |
| job | Occupation type |
| marital | Marital status |
| education | Education level |
| default | Credit default status |
| balance | Average yearly balance |
| housing | Housing loan status |
| loan | Personal loan status |
| contact | Contact communication type |
| campaign | Number of contacts during campaign |
| pdays | Days since last contact |
| previous | Number of contacts before campaign |

### Target Variable:
- **y** → Model output indicating whether customer subscribed to deposit (`yes/no`)

#  Exploratory Data Analysis (EDA)
- 45,211 records
##  Missing Value Check
The dataset was clean and contained **no missing values**, which makes it suitable for direct modeling.

##  Categorical Feature Diversity
The dataset contained several categorical variables with manageable unique values such as:
- `job` (11 unique)
- `marital` (3 unique)
- `education` (3 unique)
- `contact` (3 unique)

---

#  Feature Engineering & Preprocessing

## 1️ Encoding Categorical Features
Categorical variables were encoded using OneHotEncoding inside a **ColumnTransformer pipeline**.:

- job
- marital
- education
- default
- housing
- loan
- contact

## 2️ ML Pipeline
A **ColumnTransformer** was built using a pipeline to ensure consistent preprocessing and prediction.:
- **XGBoost Classifier**
The final model is XGBoost, a powerful gradient boosting algorithm for tabular data.

## 3️ Train-Test Split
The dataset was split into:

- **75% Training**
- **20% Testing**

This ensures the model is evaluated on unseen data and reduces overfitting risk.

#  Machine Learning Models Built
- XGBoost
- Key Parameters used :
```
 n_estimators = 300
max_depth = 7
learning_rate = 0.1
subsample = 0.5
colsample_bytree = 0.8
max_leaf_nodes = 1000
scale_pos_weight = class_0 / class_1
random_state = 42 
```
Each model was evaluated using classification metrics rather than only accuracy.
- Prediction = Yes if Probability > 0.65
---

#  Model Evaluation Metrics (Why Accuracy Alone is Misleading)

This is a **highly imbalanced classification problem**, where:
- Majority class = Non-subscribers
- Minority class = Subscribers

If a model predicts only "No", it can still achieve high accuracy.
###  Precision
> Out of predicted "Yes", how many were actually Yes?
Useful when false positives are costly (wasting marketing calls).

###  Recall
> Out of actual Yes, how many were correctly identified?
Important when missing a subscriber is costly (lost opportunity).

###  F1-Score
> Harmonic mean of precision and recall.
Best for imbalanced datasets.

###  ROC-AUC Score
> Measures ability to separate classes across probability thresholds.
Higher AUC indicates better class discrimination.


# 📊 Model Performance Comparison

## XGBoost Classifier 
XGBoost was selected as the base model

### Why Random Forest ?
Random Forest improves Decision Trees by:
- Creating multiple trees (ensemble learning)
- Reducing overfitting using weak learners
- Capturing complex feature interactions

Most importantly:
XGBoost Classifier produced the best balance between:
- accuracy
- precision
- recall
- F1-score

This makes it more reliable for real-world marketing decision-making.

##  XGBoost Evaluation Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Accuracy | ~0.85 | Overall predictions were correct for most customers |
| Precision | ~0.64 | Out of predicted subscribers, 64% were actual subscribers |
| Recall | ~0.63 | Captured around 63% of actual subscribers |
| F1-Score | ~0.64 | Balanced performance between precision & recall |

###  Key Interpretation
XGBoost provides a strong business advantage:

- It reduces false marketing calls compared to Logistic Regression & Random Forest.
- It provides better targeting efficiency.
- It maintains strong overall classification stability.

Even though recall is not extremely high, the precision is strong, which is useful when marketing cost per call is high.

---
# 📈 ROC Curve & AUC Score

### ROC Curve
ROC curve was plotted for Random Forest using predicted probabilities.

### AUC Score (Random Forest)
- **AUC ≈ 0.726**

 Interpretation:
The model has a moderate ability to distinguish between subscribers and non-subscribers.
This also indicates that marketing outcomes are influenced by complex behavioral factors, and perfect separation is difficult.

### Precision-Recall Curve
- Precision is high for low recall levels
- Precision decreases as recall increases
- This reflects the trade-off between:
    capturing more subscribers
    avoiding false positives
- The chosen threshold (0.65) balances these trade-offs.

### Model Explainability (SHAP)
The most important predictors include:
- Customer balance
- Age
- Contact type
- Number of campaign contacts
- Previous campaign outcome
- Housing loan status

```
Customers with higher balances and positive previous campaign interactions are more likely to subscribe to term deposits.
```
# 📊 Marketing Campaign Dashboard 

Along with predictive modeling, a **Marketing Campaign Dashboard** was created to visualize campaign effectiveness and customer engagement
The dashboard provides a complete business overview of:
- customer demographics
- channel performance
- conversion trends
- website engagement
- campaign type effectiveness

### Machine Learning
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- XGBoost
- SHAP
### Models Used
- Logistic Regression
- Random Forest
- XGBoost Classifier

### Model Deployment Ready
- joblib model saving (`pipeline_xgb.pkl`)

---

# 📂 Project Workflow Summary

1. Data loading and preprocessing
2. Exploratory Data Analysis (EDA)
3. Encoding categorical features
4. Feature scaling
5. Model training and comparison
6. Evaluation using classification metrics
7. ROC-AUC evaluation
8. Precision-Recall Curve
9. Model saving for deployment

---

# 📌 Conclusion
This project delivers a complete marketing analytics solution by integrating:

 **Machine Learning classification for customer subscription prediction**  

The final XGBoost model provides the best tradeoff between precision and recall, making it suitable for campaign targeting and marketing optimization.

