import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
import requests
from io import BytesIO
from PIL import Image
import folium
from streamlit_folium import folium_static
from urllib.parse import quote
st.title("Breathe Easy: Shaping Tomorrow, One Model at a Time")

# Load the excel file
data = pd.read_excel('ByCountry.xlsx')

# Get the unique values from the "Name" column
c_group_values = data['Name'].unique()

# Create a select box widget with the unique "Name" values
Name = st.selectbox('Select Country:', c_group_values)

try:
    response = requests.get(f'https://restcountries.com/v3.1/name/{Name}')
    country_data = response.json()[0]
except:
    st.write(f'Could not fetch data for {Name}.')
    country_data = None

# Create columns for layout
col1, col2 = st.columns([1, 2])

# Display flag and demographic data in the first column
if country_data:
    col1.image(country_data['flags']["png"], caption=Name, use_column_width=True)
    col2.write(f"### Demographics for {Name}")
    col2.write(f"Population: {country_data['population']}")
    col2.write(f"Area: {country_data['area']} km²")
    col2.write(f"Capital: {country_data['capital'][0]}")
    # ... (add more demographic data as needed)
else:
    col1.write(f"No data available for {Name}")

# Display map
if country_data:
    lat, lon = country_data['latlng']

    # Create a map centered around the country coordinates
    country_map = folium.Map(location=[lat, lon], zoom_start=6)
    folium_static(country_map)
else:
    col2.write(f"No map available for {Name}")

# Filter the data based on the selected "Name" value
filtered_data = data[data['Name'] == Name]

# Get the year columns
year_columns = [col for col in data.columns if col.startswith('Y_')]

# Create a dataframe for time series plot with years as one column and values as another column
time_series_data = filtered_data.melt(id_vars=['Name'], value_vars=year_columns, var_name='Year', value_name='Value')
time_series_data['Year'] = time_series_data['Year'].str.extract('(\d+)').astype(int)

# Plot the time series data using plotly
fig = px.line(time_series_data, x='Year', y='Value', title=f'Time Series Data for {Name}')
fig.update_layout(
    xaxis = dict(
        tickmode = 'linear',
        tick0 = 1970,
        dtick = 2
    )
)
st.plotly_chart(fig)

# Prepare data for modeling

# ARIMA Model
try:
    arima_model = ARIMA(time_series_data['Value'].dropna(), order=(5,1,0))
    arima_model_fit = arima_model.fit()
    arima_forecast = arima_model_fit.forecast(steps=10)
    forecast_years = np.arange(time_series_data['Year'].max(), time_series_data['Year'].max() + 10)
    arima_forecast_df = pd.DataFrame({'Year': forecast_years, 'Value': arima_forecast})
    
    line_1 = go.Line(x=time_series_data['Year'], y=time_series_data['Value'], name="Previous Years")
    line_2 = go.Line(x=arima_forecast_df['Year'], y=arima_forecast_df['Value'], name="Prediction",  marker=dict(opacity=0))

    fig_arima = make_subplots()
    fig_arima.add_trace(line_1)
    fig_arima.add_trace(line_2)
    fig_arima.update_layout(
        title="ARIMA Prediction", 
        xaxis = dict(
            tickmode = 'linear',
            tick0 = 1970,
            dtick = 2
        )
    )
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
    forecast_years = np.arange(time_series_data['Year'].max(), time_series_data['Year'].max() + 10)
    lr_forecast_df = pd.DataFrame({'Year': forecast_years, 'Value': lr_forecast[-10:]})
    # combined_data = pd.concat([time_series_data, lr_forecast_df], ignore_index=True)
    line_1 = go.Line(x=time_series_data['Year'], y=time_series_data['Value'], name="Previous Years")
    line_2 = go.Line(x=lr_forecast_df['Year'], y=lr_forecast_df['Value'], name="Prediction",  marker=dict(opacity=0))

    fig_lr = make_subplots()
    fig_lr.add_trace(line_1)
    fig_lr.add_trace(line_2)
    fig_lr.update_layout(
        title="Linear Regression Prediction", 
        xaxis = dict(
            tickmode = 'linear',
            tick0 = 1970,
            dtick = 2
        )
    )
    
    st.plotly_chart(fig_lr)
except Exception as e:
    st.write('Error in Linear Regression model:', str(e))
