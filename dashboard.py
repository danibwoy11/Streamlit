import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def dashboard_page():
    st.title('DASHBOARD PAGE')

    #Load data
    data = pd.read_csv("data/trainset.csv")

    st.header("Data Overview")
    st.write('Here is a quick summary of dataset')
    st.dataframe(data.head())


    st.subheader("Churn Count")
    churn_count = data['Churn'].value_counts()
    st.bar_chart(churn_count)


    #Plot the data
    st.subheader('Correlation Heatmap')
    corr = data[['tenure','TotalCharges']].corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    st.pyplot(plt)