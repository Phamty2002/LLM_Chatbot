import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Replace this path with the path to your forecast_summary.csv
summary_file = r'C:\Users\Pham Ty\Desktop\Thesis-Predict\Dataset_processed_1\forecast_summary.csv'
summary_df = pd.read_csv(summary_file)

# Function to parse string representations of lists back into actual lists
def parse_str_list(str_list):
    return [float(item.strip()) for item in str_list.split(',')]

# Main function to create the dashboard
def main():
    st.title('Stock Price Prediction Dashboard')

    # Sidebar for symbol selection
    symbol_list = summary_df['Symbol'].unique()
    selected_symbol = st.sidebar.selectbox('Select a Stock Symbol', symbol_list)

    # Filter the DataFrame for the selected symbol
    symbol_data = summary_df[summary_df['Symbol'] == selected_symbol].iloc[0]

    # Parse the lists from string to actual lists
    predicted_prices = parse_str_list(symbol_data['Predicted_Prices'])
    actual_prices = parse_str_list(symbol_data['Actual_Prices'])
    future_prices = parse_str_list(symbol_data['Future_Price_Predictions'])

    # Display evaluation metrics
    st.subheader(f'Evaluation Metrics for {selected_symbol}')
    st.write(f"RMSE: {symbol_data['RMSE']:.4f}")
    st.write(f"MSE: {symbol_data['MSE']:.4f}")
    st.write(f"MAPE: {symbol_data['MAPE']:.4f}")

    # Plot Actual vs Predicted Prices on Test Set
    st.subheader('Actual vs Predicted Prices on Test Set')
    fig, ax = plt.subplots()
    ax.plot(actual_prices, label='Actual Prices')
    ax.plot(predicted_prices, label='Predicted Prices')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)

    # Plot Future Price Predictions for Next Year
    st.subheader('Future Price Predictions for Next Year')
    fig2, ax2 = plt.subplots()
    ax2.plot(future_prices, label='Predicted Future Prices')
    ax2.set_xlabel('Days Ahead')
    ax2.set_ylabel('Price')
    ax2.legend()
    st.pyplot(fig2)

if __name__ == '__main__':
    main()
