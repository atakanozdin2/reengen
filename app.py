

import pandas as pd
import streamlit as st
import datetime
import openpyxl
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")
import sklearn

# --------------------------------------------------------------------------------------------------------
import requests
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px

from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit,train_test_split
from catboost import CatBoostRegressor

def select_period(period):
    periods={"1 gün":24,"2 gün":48,"3 gün":72,"1 hafta":168,"2 hafta":336,"3 hafta":504," hafta":672}
    return periods[period]

def get_consumption_data(start_date,end_date):
    df = pd.read_excel("energy.xlsx")
    df['datetime']=pd.to_datetime(df.datetime.str[:16])
    return df

def date_features(df):
    df_c=df.copy()
    df_c['month']=df_c['datetime'].dt.month
    df_c['year']=df_c['datetime'].dt.year
    df_c['hour']=df_c['datetime'].dt.hour
    df_c['quarter']=df_c['datetime'].dt.quarter
    df_c['dayofweek']=df_c['datetime'].dt.dayofweek
    df_c['dayofyear']=df_c['datetime'].dt.dayofyear
    df_c['dayofmonth']=df_c['datetime'].dt.day
    df_c['weekofyear']=df_c['datetime'].dt.weekofyear
    return(df_c)

def rolling_features(df,fh):
    df_c=df.copy()
    rolling_windows=[fh,fh+3,fh+10,fh+15,fh+20,fh+25]
    lags=[fh,fh+5,fh+10,fh+15,fh+20,fh+30]
    for a in rolling_windows:
        df_c['rolling_mean_'+str(a)]=df_c['consumption (kWh)'].rolling(a,min_periods=1).mean().shift(1)
        df_c['rolling_std_'+str(a)]=df_c['consumption (kWh)'].rolling(a,min_periods=1).std().shift(1)
        df_c['rolling_min_'+str(a)]=df_c['consumption (kWh)'].rolling(a,min_periods=1).min().shift(1)
        df_c['rolling_max_'+str(a)]=df_c['consumption (kWh)'].rolling(a,min_periods=1).max().shift(1)
        df_c['rolling_var_'+str(a)]=df_c['consumption (kWh)'].rolling(a,min_periods=1).var().shift(1)
    for l in lags:
        df_c['consumption_lag_'+str(l)]=df_c['consumption (kWh)'].shift(l)
    return(df_c)

# --------------------------------------------------------------------------------------------------------

st.set_page_config(page_title="Öngörü Aracı")

tabs= ["Öngörü","Görselleştirme","Hakkında"]

page = st.sidebar.radio("Sekmeler",tabs)

if page == "Öngörü":
    st.markdown("<h1 style='text-align:center;'>Consumption (kWh) Öngörü Çalışması</h1>",unsafe_allow_html=True)
    st.write("""Bu sayfada tahmin uzunluğı seçilerek sonuçlar elde edilmektedir.""")
    fh_selection=st.selectbox("Tahmin uzunluğunu seçiniz(Max 4 hafta)",["1 gün","2 gün","3 gün","1 hafta","2 hafta","3 hafta","4 hafta"])
    button=st.button("Tahmin Et")

    if button==True:
        with st.spinner("Tahmin yapılıyor, lütfen bekleyiniz..."):
            start_date="2018-01-01 00:00:00"
            end_date="2020-01-31 23:00:00"
            df=get_consumption_data(start_date=str(start_date),end_date=str(end_date)).iloc[:-1]
            
            #fig1,fig2 = forecast_func(df,select_period(fh_selection))
            
            #--------------------------------------------------------------------------------------------
            df1=df.copy()
            df1['ds']=df1.datetime
            df1['y']=df1['temperature (Celsius)'] 
            df1['ds']= pd.to_datetime(df1['ds'])
            
            from prophet import Prophet
            model = Prophet()
            model.fit(df1)

            future_temp = pd.date_range("2020-01-01", periods=744, freq="H") # 744 is 31x24 hours
            future_temp = pd.DataFrame(future_temp)
            future_temp.columns = ['ds']
            future_temp['ds']= pd.to_datetime(future_temp['ds'])

            #predict model
            forecast_temp = model.predict(future_temp)
            
            #Prophet for Consumption with Temperature
            future_con = pd.date_range("2020-01-01", periods=744, freq="H") # 744 is 31x24 hours
            future_con = pd.DataFrame(future_con)
            future_con.columns = ['ds']
            future_con['ds']= pd.to_datetime(future_con['ds'])

            future_con = future_con.assign(temp=forecast_temp['yhat'])
            future_con.rename(columns={'temp': 'temperature (Celsius)'}, inplace=True)

            df2=df.copy()

            df2['ds']=df2.datetime
            df2['y']=df2['consumption (kWh)']

            modelx = Prophet()
            modelx.add_regressor('temperature (Celsius)')
            modelx.fit(df2) 
            
            forecast_con = modelx.predict(future_con)

            result =pd.DataFrame()
            result['datetime'] = forecast_con['ds']
            result['consumption (kWh)']=forecast_con['yhat']
            result['temperature (Celsius)']=future_con['temperature (Celsius)']

            result.set_index('datetime')
            df_final = df.append(result)

            import plotly.graph_objects as go

            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=df_final.datetime.iloc[0:16170],y=df_final['consumption (kWh)'].iloc[0:16170],name='Geçmiş Veriler',mode='lines'))
            fig1.add_trace(go.Scatter(x=df_final.datetime.iloc[16170:],y=df_final['consumption (kWh)'].iloc[16170:],name='Öngörü',mode='lines'))

            #--------------------------------------------------------------------------------------------

            st.markdown("<h3 style='text-align:center;'>Tahmin sonuçları</h3>",unsafe_allow_html=True)
            st.plotly_chart(fig1)

elif page == "Görselleştirme":
    st.markdown("<h1 style='text-align:center;'>Veri Görselleştirme Sekmesi</h1>",unsafe_allow_html=True)

    start_date=st.sidebar.date_input(label="Başlangıç Tarihi",value=datetime.date.today()-datetime.timedelta(days=10),max_value=datetime.date.today())
    end_date=st.sidebar.date_input(label="Bitiş Tarihi",value=datetime.date.today())
    
    df_vis = get_consumption_data(start_date=str(start_date),end_date=str(end_date))
    df_describe=pd.DataFrame(df_vis.describe())
    st.markdown("<h3 style='text-align:center;'>Tüketim-Tanımlayıcı İstatistikler</h3>",unsafe_allow_html=True)
    st.table(df_describe)

    fig3=go.Figure()
    fig3.add_trace(go.Scatter(x=df_vis.datetime,y=df_vis['consumption (kWh)'],mode='lines',name='Tüketim (kWh)'))
    fig3.update_layout(xaxis_title='Date',yaxis_title="Consumption")
    st.markdown("<h3 style='text-align:center;'>Tüketim (kWh)</h3>",unsafe_allow_html=True)
    st.plotly_chart(fig3)

elif page == "Hakkında":
    st.header("Model Hakkında")
    st.write("Bu Yapay Zeka Modeli Atakan Özdin tarafından Reengen için hazırlanmıştır.")
    st.write("v.0.1")
    st.markdown("""**[Atakan Özdin](https://tr.linkedin.com/in/atakanozdin)** """)