

import pandas as pd
import streamlit as st
import datetime
import openpyxl
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

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
    periods={"1 gün":24,"2 gün":48,"3 gün":72,"1 hafta":168,"2 hafta":336}
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

def forecast_func(df,fh):
    fh_new=fh+1
    date=pd.date_range(start=pd.to_datetime(df.date).tail(1).iloc[0], periods=fh_new, freq='H')
    date=pd.DataFrame(date).rename(columns={0:"date"})
    df_fe=pd.merge(df,date,how='outer')

    #feature engineering
    df_fe=rolling_features(df_fe,fh_new)
    df_fe=date_features(df_fe)
    df_fe=df_fe.iloc[fh_new+30:].reset_index(drop=True)

    #train/test split
    split_date = pd.to_datetime(df_fe.date).tail(fh_new).iloc[0]
    print(split_date)
    historical = df_fe.loc[df_fe.date < split_date]
    y=historical[['datetime','consumption (kWh)']].set_index('datetime')
    X=historical.drop('consumption (kWh)',axis=1).set_index('datetime')
    forecast_df=df_fe.loc[df_fe.datetime > split_date].set_index('datetime').drop('consumption (kWh)',axis=1)


    tscv = TimeSeriesSplit(n_splits=3,test_size=fh_new)
    
    score_list = []
    fold = 1
    unseen_preds = []
    importance = []
    #cross validation step
    for train_index, test_index in tscv.split(X,y):
        X_train, X_val = X.iloc[train_index], X.iloc[test_index]
        y_train, y_val = y.iloc[train_index], y.iloc[test_index]
        print(X_train.shape,X_val.shape)

        cat = CatBoostRegressor(iterations=1000,eval_metric='MAE',allow_writing_files=False)
        cat.fit(X_train,y_train,eval_set=[(X_val,y_val)],early_stopping_rounds=150,verbose=50)

        forecast_predicted=cat.predict(forecast_df)
        unseen_preds.append(forecast_predicted)

        score = mean_absolute_error(y_val,cat.predict(X_val))
        print(f"MAE Fold-{fold} : {score}")
        score_list.append(score)
        importance.append(cat.get_feature_importance())
        fold+=1
    print("CV Mean Score:",np.mean(score_list))
    print(r2_score(y_val,cat.predict(X_val)))

    forecasted=pd.DataFrame(unseen_preds[2],columns=['forecasting']).set_index(forecast_df.index)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df_fe.date.iloc[-fh_new*5:],y=df_fe.consumption.iloc[-fh_new*5:],name='Tarihsel Veri',mode='lines'))
    fig1.add_trace(go.Scatter(x=forecasted.index,y=forecasted["forecasting"],name='Öngörü',mode='lines'))
    fig1.update_layout(legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
    f_importance=pd.concat([pd.Series(X.columns.to_list(),name="Feature"),pd.Series(np.mean(importance,axis=0),name="Importance")],axis=1).sort_values(by="Importance",ascending=True)
    fig2 = px.bar(f_importance.tail(10), x='Importance', y='Feature')
    return fig1,fig2
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
            start_date="2020-01-01 00:00:00"
            end_date="2020-01-31 23:00:00"
            df=get_consumption_data(start_date=str(start_date),end_date=str(end_date)).iloc[:-1]
            fig1,fig2 = forecast_func(df,select_period(fh_selection))
            st.markdown("<h3 style='text-align:center;'>Tahmin sonuçları</h3>",unsafe_allow_html=True)
            st.plotly_chart(fig1)
            st.markdown("<h3 style='text-align:center;'>Model için en önemli değişkenler</h3>",unsafe_allow_html=True)
            st.plotly_chart(fig2)

elif page == "Görselleştirme":
    st.markdown("<h1 style='text-align:center;'>Veri Görselleştirme Sekmesi</h1>",unsafe_allow_html=True)
    start_date=st.sidebar.date_input(label="Başlangıç Tarihi",value=datetime.date.today()-datetime.timedelta(days=10),max_value=datetime.date.today())
    end_date=st.sidebar.date_input(label="Bitiş Tarihi",value=datetime.date.today())
    df_vis = get_consumption_data(start_date=str(start_date),end_date=str(end_date))
    df_describe=pd.DataFrame(df_vis.describe())
    st.markdown("<h3 style='text-align:center;'>Tüketim-Tanımlayıcı İstatistikler</h3>",unsafe_allow_html=True)
    st.table(df_describe)

    fig3=go.Figure()
    fig3.add_trace(go.Scatter(x=df_vis.date,y=df_vis.consumption,mode='lines',name='Tüketim (MWh)'))
    fig3.update_layout(xaxis_title='Date',yaxis_title="Consumption")
    st.markdown("<h3 style='text-align:center;'>Saatlik Tüketim (MWh)</h3>",unsafe_allow_html=True)
    st.plotly_chart(fig3)

elif page == "Hakkında":
    st.header("Model Hakkında")
    st.write("Bu Yapay Zeka Modeli Atakan Özdin tarafından Reengen için hazırlanmıştır.")
    st.write("v.0.1")
    st.markdown("""**[Atakan Özdin](https://tr.linkedin.com/in/atakanozdin)** """)