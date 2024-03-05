import streamlit as st
import pandas as pd
from joblib import load
import optuna
import os
from model import objective, apply_thresholds
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from lightgbm import LGBMClassifier

import warnings
warnings.filterwarnings("ignore")

model_directory = 'model'
model_path = os.path.join(model_directory, 'tip_prediction_ObesityRisk.joblib')

def predict_ObesityRisk(model_path, Gender, Age, Height, Weight, family_history_with_overweight, FAVC, FCVC, NCP, CAEC,
       SMOKE, CH2O, SCC, FAF, TUE, CALC, MTRANS):
    
    # 모델 불러오기
    pred = load(model_path)
    
    # 예측 데이터 생성
    df = pd.DataFrame([{'Gender': Gender, 'Age': Age, 'Height': Height, 'Weight': Weight, 'family_history_with_overweight': family_history_with_overweight,
                        'FAVC': FAVC, 'FCVC': FCVC, 'NCP': NCP, 'CAEC': CAEC, 'SMOKE': SMOKE, 'CH2O': CH2O, 'SCC': SCC,
                        'FAF': FAF, 'TUE': TUE, 'CALC': CALC, 'MTRANS': MTRANS}])
    
    num_cols = list(df.select_dtypes(exclude=['object']).columns)
    col_name = list(df.select_dtypes(include=['object']).columns)

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    #  object datatype columns encoding:
    labelencoder = LabelEncoder()
    object_columns = df.select_dtypes(include='object').columns.difference(['NObeyesdad'])

    for col_name in object_columns:
        if df[col_name].dtypes=='object':
            df[col_name]=labelencoder.fit_transform(df[col_name])

    # 예측값 생성
    pred_lgb = pred.predict(df)
    pred_proba = pred.predict_proba(df)
    pred_lgb = pred_lgb
    pred_proba = pred_proba  # Example: replace with actual y_pred_proba
    y_val = y_val  # Example: replace with actual y_val

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    # # Get the best thresholds
    best_thresholds = study.best_params

    test_label = pred.predict_proba(df)
    test_label = apply_thresholds(test_label, best_thresholds)
    pred_result = labelencoder.inverse_transform(test_label)

    prediction = pred_result.predict(df)
    return prediction[0]

def main():

    st.title('ObesityRisk Prediction model')
    st.write('Obesity risk prediction by considering various factors')

    Gender = st.selectbox('Gender', ['Male', 'Female'])
    Age =  st.number_input('Age', min_value=14)
    Height = st.number_input('Height', min_value=1.45)
    Weight = st.number_input('Weight', min_value=39)
    family_history_with_overweight = st.selectbox('family_history_with_overweight', ['yes', 'no'])
    FAVC = st.selectbox('FAVC', ['yes', 'no'])
    FCVC = st.number_input('FCVC', min_value=1)
    NCP = st.number_input('NCP', min_value=1)
    CAEC = st.selectbox('CAEC', ['Sometimes', 'no', 'Frequently', 'Always'])
    SMOKE = st.selectbox('SMOKE', ['yes', 'no'])
    CH2O = st.number_input('CH2O', min_value=1)
    SCC = st.selectbox('SCC', ['yes', 'no'])
    FAF = st.number_input('FAF', min_value=0, max_value=3)
    TUE = st.number_input('TUE', min_value=0, max_value=2)
    CALC = st.selectbox('CALC', ['Sometimes', 'no', 'Frequently', 'Always'])
    MTRANS = st.selectbox('MTRANS', ['Public_Transportation', 'Automobile', 'Walking', 'Motorbike', 'Bike'])


    if st.button('Prediction of Obesity Risk'):
        result = predict_ObesityRisk(model_path, Gender, Age, Height, Weight, family_history_with_overweight, FAVC, FCVC, NCP, CAEC,
       SMOKE, CH2O, SCC, FAF, TUE, CALC, MTRANS)
        st.success(f'Prediction of Obesity Risk: {result}')

if __name__ == "__main__":
    main()