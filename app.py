import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from prophet import Prophet

# Load the excel file
data = pd.read_excel('ByCountry.xlsx')

# Get the unique values from the "C_group_IM24_sh" column
c_group_values = data['C_group_IM24_sh'].unique()

# Create a select box widget with the unique "C_group_IM24_sh" values
selected_c_group = st.selectbox('Select C Group:', c_group_values)

# Filter the data based on the selected "C_group_IM24_sh" value
filtered_data = data[data['C_group_IM24_sh'] == selected_c_group]

# Get the year columns
year_columns = [col for col in data.columns if col.startswith('Y_')]

# Create a dataframe for time series plot with years as one column and values as another column
time_series_data = filtered_data.melt(id_vars=['C_group_IM24_sh'], value_vars=year_columns, var_name='Year', value_name='Value')

# Plot the time series data using plotly
fig = px.line(time_series_data, x='Year', y='Value', title=f'Time Series Data for {selected_c_group}')
st.plotly_chart(fig)


# Prepare data for modeling
time_series_data['Year'] = time_series_data['Year'].str.extract('(\d+)').astype(int)

# ARIMA Model
try:
    arima_model = ARIMA(time_series_data['Value'].dropna(), order=(5,1,0))
    arima_model_fit = arima_model.fit()
    arima_forecast = arima_model_fit.forecast(steps=10)
    forecast_years = np.arange(time_series_data['Year'].max() + 1, time_series_data['Year'].max() + 11)
    arima_forecast_df = pd.DataFrame({'Year': forecast_years, 'Value': arima_forecast})
    combined_data = pd.concat([time_series_data, arima_forecast_df], ignore_index=True)
    
    fig_arima = px.line(combined_data, x='Year', y='Value', title='ARIMA Forecast')
    st.plotly_chart(fig_arima)
except Exception as e:
    st.write('Error in ARIMA model:', str(e))

# Linear Regression Model
try:
    lr_model = LinearRegression()
    X = time_series_data['Year'].dropna().values.reshape(-1,1)
    y = time_series_data['Value'].dropna()
    lr_model.fit(X, y)
    
    X_future = np.arange(X.min(), X.max() + 11).reshape(-1,1)
    lr_forecast = lr_model.predict(X_future)
    forecast_years = np.arange(time_series_data['Year'].max() + 1, time_series_data['Year'].max() + 11)
    lr_forecast_df = pd.DataFrame({'Year': forecast_years, 'Value': lr_forecast[-10:]})
    combined_data = pd.concat([time_series_data, lr_forecast_df], ignore_index=True)

    fig_lr = px.line(combined_data, x='Year', y='Value', title='Linear Regression Forecast')
    st.plotly_chart(fig_lr)
except Exception as e:
    st.write('Error in Linear Regression model:', str(e))
