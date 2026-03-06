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
| duration | Last contact duration |
| campaign | Number of contacts during campaign |
| pdays | Days since last contact |
| previous | Number of contacts before campaign |
| poutcome | Outcome of previous campaign |

### Target Variable:
- **y** → Model output indicating whether customer subscribed to deposit (`yes/no`)

#  Exploratory Data Analysis (EDA)

##  Missing Value Check
The dataset was clean and contained **no missing values**, which makes it suitable for direct modeling.

##  Categorical Feature Diversity
The dataset contained several categorical variables with manageable unique values such as:
- `job` (11 unique)
- `marital` (3 unique)
- `education` (3 unique)
- `contact` (3 unique)
- `poutcome` (3 unique)

This indicates the dataset is well structured and does not require heavy dimensionality reduction.

---

#  Feature Engineering & Preprocessing

## 1️ Encoding Categorical Features
Since ML models require numerical input, categorical variables were encoded using **Label Encoding**:

- job
- marital
- education
- default
- housing
- loan
- contact
- poutcome

## 2️ Feature Scaling (Standardization)
A **StandardScaler** was applied to normalize the feature space.
Scaling was critical because models like:
- **SVM**
- **Logistic Regression**
- **KNN**
are distance-based or gradient-based, and performance can degrade if variables like `balance` and `duration` dominate smaller-scale variables.

## 3️ Train-Test Split
The dataset was split into:

- **70% Training**
- **30% Testing**

This ensures the model is evaluated on unseen data and reduces overfitting risk.

#  Machine Learning Models Built
To select the best classifier, multiple models were trained and compared:
- Logistic Regression
- KNN (K-Nearest Neighbors)
- Naive Bayes
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest

Each model was evaluated using classification metrics rather than only accuracy.

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

## 1️ Logistic Regression
**Strength:**
- Simple and interpretable baseline model.

**Performance Insight:**
- High recall for the minority class.
- But precision was low for subscribers.

 Interpretation:
Logistic Regression was good at finding potential subscribers, but it also produced many false positives, meaning it may cause marketing teams to waste outreach on wrong customers.

---

## 2 Support Vector Machine (SVM)
**Strength:**
- Works well in high-dimensional feature spaces.
- Finds strong decision boundaries.

**Performance Insight:**
- High overall accuracy.
- Very low recall for deposit subscribers.

Interpretation:
Although SVM performed well overall, it struggled to capture the minority class properly. In marketing, missing true subscribers is costly, so recall is important.

Also, SVM is computationally expensive and slower for large datasets.

---

## 3 Decision Tree
**Strength:**
- Easy interpretability.
- Captures non-linear splits.

**Performance Insight:**
- Balanced performance but not strong.
- Prone to overfitting.

📌 Interpretation:
Decision Trees can learn complex rules but may overfit training data, reducing generalization.

---

## 4 Random Forest (Final Selected Model)
Random Forest was selected as the best-performing model.

### Why Random Forest was selected?
Random Forest improves Decision Trees by:
- Creating multiple trees (ensemble learning)
- Reducing overfitting using bagging
- Capturing complex feature interactions

Most importantly:
Random Forest produced the best balance between:
- accuracy
- precision
- recall
- F1-score

This makes it more reliable for real-world marketing decision-making.

##  Random Forest Evaluation Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Accuracy | ~0.90 | Overall predictions were correct for most customers |
| Precision | ~0.66 | Out of predicted subscribers, 66% were actual subscribers |
| Recall | ~0.32 | Captured around 32% of actual subscribers |
| F1-Score | ~0.43 | Balanced performance between precision & recall |

###  Key Interpretation
Random Forest provides a strong business advantage:

- It reduces false marketing calls compared to Logistic Regression.
- It provides better targeting efficiency.
- It maintains strong overall classification stability.

Even though recall is not extremely high, the precision is strong, which is useful when marketing cost per call is high.

---

#  Confusion Matrix Analysis
Confusion matrices were plotted for:
- SVM
- Random Forest

### Why Confusion Matrix is important?
It shows:
- True Positives (Correct subscribers predicted)
- False Positives (Marketing wasted)
- False Negatives (Missed subscribers)
- True Negatives (Correctly predicted non-subscribers)

📌 Business interpretation:
- False Positives = wasted campaign budget
- False Negatives = missed business opportunity

Random Forest showed a better tradeoff compared to SVM.
---
# 📈 ROC Curve & AUC Score

### ROC Curve
ROC curve was plotted for Random Forest using predicted probabilities.

### AUC Score (Random Forest)
- **AUC ≈ 0.67**

 Interpretation:
The model has a moderate ability to distinguish between subscribers and non-subscribers.
This also indicates that marketing outcomes are influenced by complex behavioral factors, and perfect separation is difficult.

# 📊 Marketing Campaign Dashboard (Business Intelligence Layer)

Along with predictive modeling, a **Marketing Campaign Dashboard** was created to visualize campaign effectiveness and customer engagement
The dashboard provides a complete business overview of:
- customer demographics
- channel performance
- conversion trends
- website engagement
- campaign type effectiveness


## 📌 Dashboard KPI Metrics (Top Summary Cards)

### ✅ Median Age = 43
📌 Insight:
Banks can prioritize campaigns around financially stable age groups.

###  Total Customers = 8K
📌 Insight:
The dataset provides a large enough sample for reliable segmentation and targeting.

### Total Conversions = 7K
📌 Insight:
High conversion count indicates strong campaign impact or effective targeting.

### Average Income = $84.7K
📌 Insight:
Customers have high purchasing power, meaning premium financial products can be promoted.

### ✅ Average Conversion Rate = 10.44%
📌 Insight:
A conversion rate above 10% is strong in financial marketing, suggesting campaigns were impactful.

###  Website Visits = 198K
📌 Insight:
Digital channels play a major role in customer decision-making.

# 📌 Dashboard Visual Insights

## 1️ Time On Site by Income
This chart shows how website engagement varies across income levels.

📌 Key Insights:
- Higher-income segments tend to spend more time on site.
- Engagement improves as income increases.
- Mid-income users show moderate engagement, likely due to price sensitivity.
- Banks should customize product offers depending on income bracket, since high-income users engage deeper.

---

## 2️ Email Metrics by Campaign Channel
This chart compares email-related performance across channels such as:
- Email
- PPC
- Referral
- SEO
- Social Media

📌 Key Insights:
- Referral channel shows the highest email engagement.
- PPC and SEO show moderate email responses.
- Social Media performs well in outreach but may not always translate into conversions.
- Referral-based campaigns generate stronger customer trust, leading to higher engagement.

---

## 3️ Conversions by Channels
This chart directly compares conversion counts across channels.

📌 Observed Results:
- Referral has the highest conversions.
- PPC is the second best channel.
- SEO, Email, and Social Media follow closely.
- Referral marketing should be expanded since it produces the strongest conversion output.

---

## 4️ Previous Purchases vs Campaign Type
This chart shows campaign effectiveness across stages:

- Awareness
- Consideration
- Retention
- Conversion

📌 Key Insights:
- Conversion campaigns have the strongest performance.
- Consideration and Retention are close competitors.
- Awareness campaigns show the lowest purchase contribution.
- Awareness campaigns bring traffic, but conversion-focused campaigns drive revenue.

---

## 5️ Campaign Type Metrics Table
This table summarizes performance metrics for each campaign type, including:

- Click Through Rate (CTR)
- Pages per Visit
- Email Opens
- Social Shares
- Time on Site

📌 Key Insights:
- Conversion campaigns generate the highest engagement and time on site.
- Awareness campaigns generate strong social shares.
- Consideration campaigns maintain strong CTR and pages per visit.

Business Meaning:
Different campaign types contribute differently:
- Awareness → engagement + reach
- Consideration → interest building
- Conversion → revenue generation
- Retention → long-term loyalty

---

## 6️ Avg Website Visits vs Social Shares (Scatter Plot)
This scatter plot shows the relationship between:

- average visits
- social sharing behavior

📌 Key Insights:
- Higher website visits generally correlate with higher social shares.
- A strong cluster indicates a consistent engaged user base.
- Some users have high visits but low shares (private decision makers).
- Some users share more with moderate visits (influencers).

Business Meaning:
Marketing should target:
- high-visit high-share users for viral campaigns
- high-visit low-share users for personalized conversion offers

---

#  Key Business Recommendations

Based on dashboard + ML results:

### ✅ Channel Strategy
- Invest more in **Referral and PPC channels**
- Improve conversion design for SEO and Email campaigns

### ✅ Campaign Type Strategy
- Conversion campaigns are most revenue-effective.
- Awareness campaigns should be optimized to push customers toward conversion funnel.

### ✅ Customer Targeting Strategy
- Use ML predictions to target only high-probability customers.
- Reduce campaign cost by avoiding low-probability customers.

### ✅ Digital Engagement Strategy
- High-income customers show higher engagement.
- Use personalized premium offers for high-income segments.

---

# 🛠 Tech Stack Used

### Data Analytics & Visualization
- Power BI / Dashboard Visualization

### Machine Learning
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

### Models Used
- Logistic Regression
- KNN
- Naive Bayes
- SVM
- Decision Tree
- Random Forest (Final Model)

### Model Deployment Ready
- joblib model saving (`rf_model.pkl`)

---

# 📂 Project Workflow Summary

1. Data loading and preprocessing
2. Exploratory Data Analysis (EDA)
3. Encoding categorical features
4. Feature scaling
5. Model training and comparison
6. Evaluation using classification metrics
7. ROC-AUC evaluation
8. Dashboard creation for campaign monitoring
9. Model saving for deployment

---

# 📌 Conclusion
This project delivers a complete marketing analytics solution by integrating:

 **Dashboard-based campaign performance insights**  
 **Machine Learning classification for customer subscription prediction**  

The final Random Forest model provides the best tradeoff between precision and recall, making it suitable for campaign targeting and marketing optimization.
The dashboard complements this by enabling business teams to understand which channels and campaign types drive the strongest engagement and conversions
