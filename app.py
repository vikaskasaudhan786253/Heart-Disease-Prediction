import streamlit as st
import pickle
import pandas as pd
import numpy as np
from randomforest import Random_Forest
from logisticregression import Logistic_Regression
from decisiontree import Decision_Tree
from metrics import Metrics

def decision_tree(values):
    dt = Decision_Tree()
    st.write("Accuracy Score : " + str(dt.accuracy()) + "%")
    st.write("Precision Score : " + str(dt.precision()) + "%")
    st.write("Recall Score : " + str(dt.recall()) + "%")
    st.write("F1 Score : " + str(dt.f1_scores()) + "%")
    st.write("ROC AUC Score : " + str(dt.roc_auc()) + "%")
    pred = dt.predict(values)
    proba = dt.predict_proba(values)
    if pred==1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")
    st.progress(proba/100)
    st.write(f"Risk Probability: {proba:.2f}%")

def random_forest(values):
    rf = Random_Forest()
    st.write("Accuracy Score : " + str(rf.accuracy()) + "%")
    st.write("Precision Score : " + str(rf.precision()) + "%")
    st.write("Recall Score : " + str(rf.recall()) + "%")
    st.write("F1 Score : " + str(rf.f1_scores()) + "%")
    st.write("ROC AUC Score : " + str(rf.roc_auc()) + "%")
    pred = rf.predict(values)
    proba = rf.predict_proba(values)
    if pred==1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")
    st.progress(proba/100)
    st.write(f"Risk Probability: {proba:.2f}%")


def logistic_regression(values):
    lr = Logistic_Regression()
    st.write("Accuracy Score : " + str(lr.accuracy()) + "%")
    st.write("Precision Score : " + str(lr.precision()) + "%")
    st.write("Recall Score : " + str(lr.recall()) + "%")
    st.write("F1 Score : " + str(lr.f1_scores()) + "%")
    st.write("ROC AUC Score : " + str(lr.roc_auc()) + "%")
    pred = lr.predict(values)
    proba = lr.predict_proba(values)
    if pred==1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")
    st.progress(proba/100)
    st.write(f"Risk Probability: {proba:.2f}%")

st.sidebar.title("Select The Values")


# Values Collection
age = st.sidebar.number_input("Select Age :",min_value=0,max_value=150,value=5,step=1,format="%d")
gender = st.sidebar.selectbox("Select Gender :",('Male','Female'))
gender_map = {'Male':1,'Female':0}
gender_value = gender_map.get(gender, None)
chest_pain_type = st.sidebar.selectbox("Select Chest Pain Type :",('Typical Angina','Atypical Angina','Non-Anginal Pain','Asymptomatic'))
chest_pain_type_map = {'Typical Angina': 1,
    'Atypical Angina': 2,
    'Non-Anginal Pain': 3,
    'Asymptomatic': 4}
chest_pain_type_value = chest_pain_type_map.get(chest_pain_type, None)
bp = st.sidebar.number_input("Select Blood Pressure :",min_value=0,max_value=300,value=120,step=1,format="%d")
cholesterol = st.sidebar.number_input("Select Cholesterol Level :",min_value=0,max_value=1000,value=200,step=1,format="%d")
bool = st.sidebar.selectbox("Select Fasting Blood Sugar Level is greater than 120 :",('Yes','No'))
bool_map = {'Yes':1,'No':0}
bool_value = bool_map.get(bool, None)
ekg = st.sidebar.selectbox("Select EKG Result :",('Normal','ST-T wave abnormality','Left ventricular hypertrophy'))
ekg_map = {'Normal':0,'ST-T wave abnormality':1,'Left ventricular hypertrophy':2}
ekg_value = ekg_map.get(ekg, None)
hr = st.sidebar.number_input("Select Maximum Heart Rate Achieved :",min_value=0,max_value=300,value=150,step=1,format="%d")
eia = st.sidebar.selectbox("Select Exercise Induced Angina :",('Yes','No'))
eia_map = {'Yes':1,'No':0}
eia_value = eia_map.get(eia, None)
st_depression_val = st.sidebar.number_input("Select ST Depression Induced by Exercise Relative to Rest :",min_value=0.0,max_value=10.0,value=1.0,step=0.1,format="%.1f")
slope_st = st.sidebar.selectbox("Select Slope of the Peak Exercise ST Segment :",('Upsloping','Flat','Downsloping'))
slope_st_map = {'Upsloping':1,'Flat':2,'Downsloping':3}
slope_st_value = slope_st_map.get(slope_st, None)
num_vessels = st.sidebar.number_input("Select Number of Major Vessels Colored by Fluoroscopy :",min_value=0,max_value=4,value=0,step=1,format="%d")
thallium = st.sidebar.selectbox("Select Thalassemia :",('Normal','Fixed Defect','Reversible Defect'))
thallium_map = {'Normal':3,'Fixed Defect':6,'Reversible Defect':7}
thallium_val = thallium_map.get(thallium, None)

# Model Comparison
st.header("Heart Disease Prediction using Machine Learning")
algorithms = ("Logistic Regression","Decision Tree", "Random Forest")
st.subheader("Model Comparison")
metric = Metrics()
model = {
    'accuracy':metric.Accuracy(),
    'precision':metric.Precision(),
    'recall':metric.Recall(),
    'f1_score':metric.F1_score(),
    'roc_auc':metric.Roc_Auc()
}
df = pd.DataFrame(model,index=list(algorithms))
st.write(df)


values = [[age,gender_value,chest_pain_type_value,bp,cholesterol,bool_value,ekg_value,hr,eia_value,st_depression_val,slope_st_value,num_vessels,thallium_val]]
selected_values = {'Age':age, 'Sex':gender, 'Chest pain type':chest_pain_type, 'BP':bp, 'Cholesterol':cholesterol, 'FBS over 120':bool,
       'EKG results':ekg, 'Max HR':hr, 'Exercise angina':eia, 'ST depression':st_depression_val,
       'Slope of ST':slope_st, 'Number of vessels fluro':num_vessels, 'Thallium':thallium}
new_df = pd.DataFrame(selected_values,index=[[0]])
selected_algo = st.sidebar.selectbox("Select a Algorithm",algorithms )
btn = st.sidebar.button("Run Algorithm")

# Running Algorithm
if btn:
    st.subheader("Selected Algorithm: " + selected_algo)
    if selected_algo == 'Logistic Regression':
        logistic_regression(values)
    elif selected_algo == 'Random Forest':
        random_forest(values)
    elif selected_algo == "Decision Tree":
        decision_tree(values)
