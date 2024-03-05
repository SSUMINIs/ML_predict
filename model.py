import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score
from joblib import dump, load
import os

import warnings
warnings.filterwarnings("ignore")

train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")
original_data = pd.read_csv("data/ObesityDataSet.csv")
sample_submission_data = pd.read_csv("data/sample_submission.csv")

train_data = train_data.drop("id", axis=1)
train_data = pd.concat([train_data, original_data], ignore_index=True)
train_data = train_data.drop_duplicates()

num_cols = list(train_data.select_dtypes(exclude=['object']).columns)
cat_cols = list(train_data.select_dtypes(include=['object']).columns)

num_cols_test = list(test_data.select_dtypes(exclude=['object']).columns)
cat_cols_test = list(test_data.select_dtypes(include=['object']).columns)

num_cols_test = [col for col in num_cols_test if col not in ['id']]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_data[num_cols] = scaler.fit_transform(train_data[num_cols])
test_data[num_cols_test] = scaler.transform(test_data[num_cols_test])

#  object datatype columns encoding:
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
object_columns = train_data.select_dtypes(include='object').columns.difference(['NObeyesdad'])

for col_name in object_columns:
    if train_data[col_name].dtypes=='object':
        train_data[col_name]=labelencoder.fit_transform(train_data[col_name])
        
for col_name in test_data.columns:
    if test_data[col_name].dtypes=='object':
        test_data[col_name]=labelencoder.fit_transform(test_data[col_name])

X = train_data.drop(['NObeyesdad'], axis=1)
y = train_data['NObeyesdad']
y = labelencoder.fit_transform(y)
X_test = test_data.drop(["id"],axis=1)

X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2,random_state=42)

param = {"objective": "multiclass",          
    "metric": "multi_logloss",          
    "verbosity": -1,                    
    "boosting_type": "gbdt",            
    "random_state": 42,       
    "num_class": 7,                     
    'learning_rate': 0.030962211546832760,  
    'n_estimators': 500,                
    'lambda_l1': 0.009667446568254372,  
    'lambda_l2': 0.04018641437301800,   
    'max_depth': 10,                    
    'colsample_bytree': 0.40977129346872643,  
    'subsample': 0.9535797422450176,   
    'min_child_samples': 26}

model_lgb = lgb.LGBMClassifier(**param,verbose=100)
pred = model_lgb.fit(X_train, y_train)

def objective(trial):
    # Define the thresholds for each class
    thresholds = {}
    for i in range(num_classes):
        thresholds[f'threshold_{i}'] = trial.suggest_uniform(f'threshold_{i}', 0.0, 1.0)

    # Apply the thresholds to convert probabilities to predictions
    y_pred = apply_thresholds(pred_proba, thresholds)

    # Calculate accuracy
    accuracy = accuracy_score(y_val, y_pred)
    return accuracy  

def apply_thresholds(y_proba, thresholds):
    # Apply the specified thresholds to convert probabilities to predicted labels
    y_pred_labels = np.argmax(y_proba, axis=1)
    for i in range(y_proba.shape[1]):
        y_pred_labels[y_proba[:, i] > thresholds[f'threshold_{i}']] = i

    return y_pred_labels


model_directory = 'model'

if not os.path.exists(model_directory):
    os.makedirs(model_directory)

model_path = os.path.join(model_directory, 'tip_prediction_ObesityRisk.joblib')
dump(pred, model_path)