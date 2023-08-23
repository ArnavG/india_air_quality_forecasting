import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os import listdir
from os.path import isfile, join

import streamlit as st

import statsmodels.api as sm

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

from pmdarima.arima import auto_arima
from pandas.tseries.offsets import DateOffset
from datetime import datetime, timedelta

import plotly.express as px

import warnings
warnings.filterwarnings("ignore")

PATH = "AirQualityIndia/archive"

def get_states():
    states = []
    for f in listdir(PATH):
        if f == "stations_info.csv" or f == ".DS_Store":
            continue
        else:
            states.append(f[0:2])
    return list(set(states))

def state_data(state):
    states = get_states()
    if state not in states:
        return "Invalid state code"
    else:
        state_df = pd.DataFrame()
        for f in listdir(PATH):
            if f == "stations_info.csv" or f == ".DS_Store":
                continue
            else:
                if f[0:2] == state:
                    df = pd.read_csv(f"AirQualityIndia/archive/{f}")
                    state_df = pd.concat([state_df, df])
        return state_df
    
def to_monthly_avg(df):
    df["From Date"] = pd.to_datetime(df["From Date"])
    df["Year"] = ((df["From Date"]).dt.year).astype(int)
    df["Month"] = ((df["From Date"]).dt.month).astype(int)
    monthly_data = df.drop(["From Date", "To Date"], axis=1)
    monthly_data = monthly_data.groupby(["Year", "Month"], as_index=False).agg(np.mean)
    monthly_data = monthly_data.set_index(pd.to_datetime(monthly_data[['Year','Month']].assign(day=1)))
    monthly_data["Date"] = monthly_data.index
    
    def full_cols_only(monthly_df):
        mdf = pd.DataFrame()
        mdf.index = monthly_df.index
        for col in monthly_df.columns:
            if not monthly_df[col].hasnans:
                mdf[col] = monthly_df[col]
        return mdf
    
    return full_cols_only(monthly_data)

# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    diff.append(0)
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return diff

# invert differenced forecast
def inverse_difference(last_ob, value):
    return value + last_ob

num_diffs = 0
def check_stationary(df, col):
    p_val = adfuller(df[col])[1]
    if p_val < 0.05:
        return df
    else:
        df["diff"] = difference(df[col])
        global num_diffs
        num_diffs += 1
        return check_stationary(df, "diff")
    
def inverse_difference(last_ob, value):
    return value + last_ob

def aq_sarima(df, col, state):
    
    original_var = col
    
    diff_in_cols = False
    
    if "diff" in df.columns:
        col = "diff"
        diff_in_cols = True
    
    features = df.drop(["Year", "Month", "Date"], axis=1)
    y_var = df[col]
    # ARIMA parameters
    
    d_param = 1
    D_param = 1

    stepwise_model = auto_arima(y_var, X = features.drop(col, axis=1), start_p=0, start_q=0,
                               max_p=3, max_q=3, m=12,
                               start_P=0, seasonal=True,
                               d=d_param, D=D_param, trace=False,
                               error_action='ignore',  
                               suppress_warnings=True, 
                               stepwise=True)
    
    train_bound = int(np.round(0.7 * len(features)))
    train = features.iloc[0:train_bound]
    test = features.iloc[(train_bound + 1):]
    len_original = len(test[original_var])
    

    future_dates = pd.date_range(start = test.index[-1], end = test.index[-1] + timedelta(365 * 3), freq="MS")
    future_datest_df = pd.DataFrame(index=future_dates[1:], columns=test.columns)
    test = pd.concat([test, future_datest_df], axis=0)
    
    stepwise_model.fit(train[col])
    
    future_forecast = stepwise_model.predict(n_periods=len(test) + 1)
    test["Prediction"] = future_forecast
    
    if not diff_in_cols:
        test["inverted"] = test["Prediction"]
        if state == "KA":
            test["inverted"] = test["inverted"] - np.mean(np.abs(test["inverted"].head(47) - test["CO (mg/m3)"].head(47)))
        return test
    else:
        # define a dataset with a linear trend
        data = list(test[original_var].dropna(axis=0))
        diff = list(test["diff"].dropna(axis=0))
        preds = list(test["Prediction"])

        # invert the difference
        inverted = []
        inverted.append(data[0])

        # known values
        for i in range(1, len_original):
            invd = inverse_difference(preds[i], data[i - 1])
            inverted.append(invd)

        # forecasted values
        for j in range(len_original, len(preds)):
            last_val = data[-1]
            invd = inverse_difference(preds[j], last_val)
            data.append(invd)
            inverted.append(invd)

        test["inverted"] = inverted
        return test

@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')


def main():
    st.title("Forecasting Carbon Monoxide (CO) Emissions in Indian States using SARIMAX")
    option = st.selectbox(
    "Select a state.",
    ("Andhra Pradesh", "Karnataka", "Uttar Pradesh", "West Bengal"),
    placeholder = "")
    states_dict = {"Andhra Pradesh": "AP",
                   "Karnataka": "KA",
                   "Uttar Pradesh": "UP",
                   "West Bengal": "WB"
                   }
    state = states_dict[option]
    st.write('You selected:', option)
    aq_data = state_data(state)
    aq_data = to_monthly_avg(aq_data)
    aq_data = check_stationary(aq_data, "CO (mg/m3)")
    d = num_diffs
    aq_data
    csv = convert_df(aq_data)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name=f"air_quality_data_{state}",
    )


    df_test = aq_sarima(aq_data, "CO (mg/m3)", state)

    df_first_period = pd.DataFrame(aq_data["CO (mg/m3)"]).dropna()
    first_period = pd.Series(["Actual Trend"]).repeat(len(df_first_period))
    first_period.index = df_first_period.index
    df_first_period["period"] = first_period

    df_second_period = pd.DataFrame(df_test["inverted"]).rename({"inverted": "CO (mg/m3)"}, axis="columns")
    second_period = pd.Series(["Forecasted Emissions"]).repeat(len(df_second_period))
    second_period.index = df_second_period.index
    df_second_period["period"] = second_period

    df_all_periods = pd.concat([df_first_period, df_second_period])
    fig = px.line(df_all_periods, x=df_all_periods.index, y="CO (mg/m3)", color="period")
    st.plotly_chart(fig)






# Using the special variable 
# __name__
if __name__=="__main__":
    main()