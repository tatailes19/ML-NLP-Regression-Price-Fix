import streamlit as st
import pandas as pd
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.feature_extraction.text import CountVectorizer

# Function to process data (your code logic)
def process_data(file):
    try:
        # Read the Excel file using Polars (avoid converting to Pandas unless necessary)
        data = pl.read_excel(file)
        
        # Filter out 'test' entries directly in Polars
        df = data.filter(data["Username"] != "test").to_pandas()

        # Vectorized string operations to avoid apply()
        df["Model"] = df["Brand"] + " " + df["Product"] + " " + df["RAM"].astype(str) + " " + df["STOCK"].astype(str) + " " + df["Source"]

        # Calculate mode prices for each model
        mode_prices = df.groupby('Model')['Sell Price'].agg(lambda x: x.mode().iloc[0]).reset_index()
        df = pd.merge(df, mode_prices, on='Model', how='left', suffixes=('', '_mode'))

        # Calculate the absolute and percentage difference
        df['abs_diff'] = abs(df['Sell Price'] - df['Sell Price_mode'])
        df['perc_diff'] = (df['abs_diff'] / df['Sell Price_mode']) * 100

        # Flag anomalies where the percentage difference is more than 15%
        df['status'] = df['perc_diff'].apply(lambda x: 'anomaly' if x > 15 else 'normal')
        count = df['status'].value_counts()

        # Separate anomaly and normal data
        anomaly_data = df[df['status'] == 'anomaly'][["Model", "Sell Price"]]
        normal_data = df[df['status'] == 'normal'][["Model", "Sell Price"]]

        # Prepare data for model training
        X_normal = normal_data['Model']
        y_normal = normal_data['Sell Price']
        X_anomaly = anomaly_data['Model']
        y_anomaly = anomaly_data['Sell Price']

        # Scale the 'Sell Price' values
        scaler = StandardScaler()
        y_normal_scaled = scaler.fit_transform(y_normal.values.reshape(-1, 1)).ravel()

        # Vectorize the model names (features)
        vectorizer = CountVectorizer(stop_words=None, max_features=2000, min_df=1, max_df=0.85, token_pattern=r'\b\w+\b')
        X_normal_vectorized = vectorizer.fit_transform(X_normal)
        X_anomaly_vectorized = vectorizer.transform(X_anomaly)

        # Train a model on the normal data
        X_train, X_test, y_train, y_test = train_test_split(X_normal_vectorized, y_normal_scaled, test_size=0.00001, random_state=42)

        # Train an XGBoost model
        model = xgb.XGBRegressor(n_estimators=100, random_state=42, tree_method='hist', max_depth=5)
        model.fit(X_train, y_train)

        # Predict the prices for anomalies
        y_anomaly_pred_scaled = model.predict(X_anomaly_vectorized)

        # Add noise (as per original code)
        noise = np.random.normal(0, 0.02, size=y_anomaly_pred_scaled.shape)
        y_anomaly_pred_scaled = y_anomaly_pred_scaled + noise

        # Inverse transform the predicted values to get them back to the original scale
        y_anomaly_pred = scaler.inverse_transform(y_anomaly_pred_scaled.reshape(-1, 1))

        # Round the predicted prices
        rounded_pred_prices = np.round(y_anomaly_pred / 100) * 100

        # Update the anomalies in the original dataframe with the predicted values
        df.loc[df['status'] == 'anomaly', 'Sell Price'] = rounded_pred_prices

        # Clean up the dataframe by dropping unnecessary columns in-place
        df.drop(columns=["Sell Price_mode", "abs_diff", "perc_diff", "status", "Model"], inplace=True)

        # Return the cleaned data and status counts
        return df, count

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None, None

# Streamlit UI
def main():
    st.title('🔧 Anomaly Prices Fixing Tool')
    st.subheader('📊 Upload your data file to fix anomalies in the sell prices.')

    st.markdown("""
    - This tool processes data files to detect anomalies in sell prices based on the differences from the mode prices. 🕵️‍♂️
    - A model is used to predict the correct sell prices for these anomalies. 🧑‍💻
    - After processing, you can download the cleaned data file with fixed prices. 📥
    """)
    
    st.info("Ensure that your file has the columns: 'Username', 'Brand', 'Product', 'RAM', 'STOCK', 'Source', and 'Sell Price'. 🔑")

    # Upload file
    uploaded_file = st.file_uploader("Choose an Excel file 📂", type=["xlsx"])
    
    if uploaded_file is not None:
        st.write("✅ File uploaded successfully! Processing... Please wait.")
        
        # Show a progress bar while processing
        with st.spinner("Understanding Your Data... 📈"):
            df_fixed, status_counts = process_data(uploaded_file)
        
        # Once the data is processed, display it for download
        if df_fixed is not None:
            st.success("🎉 Data Analyzed successfully!")
            st.write("Here is a preview of the data:")

            # Show the first few rows of the fixed data
            st.dataframe(df_fixed.head())
            
            # Plot frequency of 'normal' and 'anomaly'
            st.subheader('📊 Frequency of Anomalies and Normal Data')
            fig, ax = plt.subplots()
            status_counts.plot(kind='bar', ax=ax, color=['green', 'red'])
            ax.set_title('Normal vs Anomalies in Sell Price')
            ax.set_xlabel('Status')
            ax.set_ylabel('Frequency')
            st.pyplot(fig)
            
            # Now show the "Preparing to download" spinner until the download button appears
            with st.spinner("Preparing to download... ⬇️"):
                # Save the file after processing as CSV
                output_file = "kpis_fixed.csv"
                df_fixed.to_csv(output_file, index=False)
            
                # The spinner stops here, and the download button appears right after
                with open(output_file, "rb") as f:
                    st.success("🎉 Data Ready to Download!")
                    st.download_button(
                        label="Download Fixed Data 📥",
                        data=f,
                        file_name=output_file,
                        mime="text/csv"
                    )
        else:
            st.error("❌ Sorry, there was an error while processing the file.")

if __name__ == "__main__":
    main()
