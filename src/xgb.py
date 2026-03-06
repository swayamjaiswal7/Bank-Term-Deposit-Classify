import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("./data/bank-marketing-cleaned.csv"
                   ,usecols=['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
       'loan', 'contact', 'campaign', 'pdays', 'previous'])
y = pd.read_csv("./data/outcome.csv",usecols=['y']).squeeze()

def data_trf(X,y,test_size=0.25):
    '''Train Test Split
    Label Encoding'''
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size,random_state=42,stratify=y)
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    return X_train,X_test,y_train_enc,y_test_enc

X_train,X_test,y_train,y_test = data_trf(data,y)

def column_transformer(cat_cols:list,num_cols:list):
    '''Applying One hot Encoding or Scaling'''
    transformers = ColumnTransformer(transformers=[
        ('cat',OneHotEncoder(handle_unknown='ignore'),cat_cols),
        ('num','passthrough',num_cols)
    ])
    return transformers

transformers = column_transformer(['job','marital','education','default','housing','loan','contact'],
                                  ['age','balance','campaign','pdays','previous'])


scale_pos_ = len(y_train[y_train==0]) / len(y_train[y_train==1])
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.1,
    subsample=0.5,
    colsample_bytree=0.8,
    booster='gbtree',
    scale_pos_weight=scale_pos_,
    eval_metric='logloss',
    random_state=42
)
def pipeline_train(encoder,model,X_train,X_test,y_train,y_test):
    pipe_xgb = Pipeline([
        ('enc',encoder), #encoding
        ('xgb',model) # model
        ])
    pipe_xgb.fit(X_train,y_train)
    y_pred = pipe_xgb.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    print(classification_report(y_test,y_pred))

    return accuracy ,pipe_xgb

accuracy,pipe_xbg =pipeline_train(transformers,xgb,X_train,X_test,y_train,y_test)
print("Accuracy of XGBoost model is ",accuracy)