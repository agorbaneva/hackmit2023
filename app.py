import streamlit as st
import pandas as pd
import plotly.express as px

# Load the excel file
data = pd.read_excel('path/to/your/excel/file.xlsx')

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
