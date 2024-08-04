import streamlit as st
import pandas as pd

df = pd.read_csv('insurance.csv')

st.write("Hello, Streamlit!")

number = st.slider("Pick a number", 0, 100)

seats = selectbox("enter the num of seats : ",
                  [4,5,6,7,8])


st.write("""
# DataFrame Column Selectors
""")

# Iterate through each column in the DataFrame
for column in df.columns:
    # Create a selectbox for each column
    column.selectbox(f'Select value for {column}', options=df[column].unique(), key=column)
