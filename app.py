import streamlit as st
import pandas as pd
import joblib
model=joblib.load('logist_heartdis.pkl')
scaler=joblib.load('scalerlo.pkl')
col=joblib.load('coloflo.pkl')
if isinstance(col[0], list):
    col=col[0]

st.title("HEART STROKE PREDICTION AND ANALYSIS")
st.markdown("PROVIDE THE FOLLOWING DETAILS")
age=st.slider('(AGE',18,100,40)
se=st.selectbox('SEX',['M','F'])
chest=st.selectbox('CHESTPAINTYPE',['ATA','NAP','TA','ASY'])
rest=st.number_input('RESTING BLOOD PRESSURE(mm Hg)',80,200,120)
fasting=st.selectbox('FASTING BLOOD SUGAR > 120mg/dL',[0,1])
resting_ecg=st.selectbox("RESTING ECG",['Normal','ST','LVH'])
max_hr=st.slider("MAX HEART RATE",60,220,150)
exerc=st.selectbox("EXERCISE-INDUCED ANGINA",['Y','N'])
oldpick=st.slider("OLD PEAK (ST DEPRESSION)",0.0,0.6,1.0)
st_slp=st.selectbox("ST-SLOPE",['UP','FLAT','DOWN'])
if st.button("PREDICT"):
    raw_input={
        'Age':age,
        'RestingBp':rest,
        'Sex_M':1 if se=="M" else 0,
        'ChestPainType' +chest:1,
        'FastingBs':fasting,
        'RestingEcg_ST':1 if resting_ecg=='ST'else 0,
        'ExerciseAngina_Y':1 if exerc=='Y' else 0,
        'OldPeak':oldpick,
        'ST_Slope_Up':1 if st_slp=="UP" else 0,
        'ST_Slope_Flat':1 if st_slp=="FLAT" else 0

    }
    input_df=pd.DataFrame([raw_input])
    input_df=input_df.reindex(columns=col,fill_value=0)

    scaler_input=scaler.transform(input_df) 
  
    prediction=model.predict(scaler_input)[0]
    if prediction==1:
        st.error("HIGH RISK OF HEART DISEASE")
    else:
        st.error("LOW RISK OF HEART DISEASE")     

