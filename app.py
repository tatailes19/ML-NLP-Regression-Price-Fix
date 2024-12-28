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
        data = pl.read_excel(file)
        df = data.to_pandas()

        df = df[df["Username"] != "test"]
        df["Model"] = df.apply(lambda x: " ".join([x["Brand"], x["Product"], str(x["RAM"]), str(x["STOCK"]), x["Source"]]), axis=1)

        mode_prices = df.groupby('Model')['Sell Price'].apply(lambda x: x.mode().iloc[0]).reset_index()
        df = df.merge(mode_prices, on='Model', suffixes=('', '_mode'))

        df['abs_diff'] = abs(df['Sell Price'] - df['Sell Price_mode'])
        df['perc_diff'] = (df['abs_diff'] / df['Sell Price_mode']) * 100

        df['status'] = df['perc_diff'].apply(lambda x: 'anomaly' if x > 15 else 'normal')
        count = df['status'].value_counts()

        anomaly_data = df[df['status'] == 'anomaly'][["Model", "Sell Price"]]
        normal_data = df[df['status'] == 'normal'][["Model", "Sell Price"]]

        X_normal = normal_data['Model']
        y_normal = normal_data['Sell Price']
        X_anomaly = anomaly_data['Model']
        y_anomaly = anomaly_data['Sell Price']

        scaler = StandardScaler()
        y_normal_scaled = scaler.fit_transform(y_normal.values.reshape(-1, 1)).ravel()

        vectorizer = CountVectorizer(stop_words=None, max_features=5000, min_df=1, max_df=0.95, token_pattern=r'\b\w+\b')
        X_normal_vectorized = vectorizer.fit_transform(X_normal)
        X_anomaly_vectorized = vectorizer.transform(X_anomaly)

        X_train, X_test, y_train, y_test = train_test_split(X_normal_vectorized, y_normal_scaled, test_size=0.000001, random_state=42)

        model = xgb.XGBRegressor(n_estimators=300, random_state=42)
        model.fit(X_train, y_train)

        y_anomaly_pred_scaled = model.predict(X_anomaly_vectorized)
        noise = np.random.normal(0, 0.02, size=y_anomaly_pred_scaled.shape)
        y_anomaly_pred_scaled = y_anomaly_pred_scaled + noise
        y_anomaly_pred = scaler.inverse_transform(y_anomaly_pred_scaled.reshape(-1, 1))

        rounded_pred_prices = np.round(y_anomaly_pred / 100) * 100
        df.loc[df['status'] == 'anomaly', 'Sell Price'] = rounded_pred_prices
        anomaly_data['Predicted_Selling_Price'] = rounded_pred_prices
        anomaly_data['Predicted_Selling_Price'] = anomaly_data['Predicted_Selling_Price'].apply(lambda x:str(x) + "  DZD")
        anomaly_data["Sell Price"] = anomaly_data["Sell Price"].apply(lambda x: str(x) + "  DZD")

        df = df.drop(columns=["Sell Price_mode", "abs_diff", "perc_diff", "status", "Model"])

        return df, count, anomaly_data
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None, None

def main():
    st.set_page_config(page_title="Anomaly Fixing Tool",page_icon="Orbi_logo.png")#, layout="wide"

    # Display logo and title
    st.sidebar.image("logo.jpeg")

    # Center and style the sidebar title
    st.sidebar.markdown("<h1 style='text-align: center; font-family: serif; font-weight: bold; font-size: 34px;'>Orbicore</h1>", unsafe_allow_html=True)

    # Welcome message
    st.sidebar.markdown("<h1 style='text-align: center; font-family: serif;'>Welcome to our Price Anomaly fixing tool! 🛠️</h1>", unsafe_allow_html=True)

    # Sidebar menu
    menu = [":iphone: Project HHP", ":tv: Project CE"]
    choice = st.sidebar.radio("Menu", menu)

    if choice == ":iphone: Project HHP":
        st.markdown("<h1 style='text-align: center;font-family: serif'>🔧 HHP Data Fixing Tool 📱</h1>", unsafe_allow_html=True)

        st.markdown("<h2 style='text-align: center;font-family: serif'>⌚ A simple tool that helps you deal with abnormal prices within the KPIs Dataset using ML 🚀</h2>", unsafe_allow_html=True)

        # Using a container for the main description
        with st.container():
            st.markdown("""
            - Upload an Excel file and the tool processes data files to detect anomalies in sell prices based on the differences from the mode prices. 🕵️‍♂️
            - An Xgboost model is then used to predict the correct sell prices for these anomalies. 🧑‍💻
            - After processing, you can download the cleaned data file with fixed prices. 📥
            """)

        st.markdown("---")  # Horizontal line for separation

        # Using columns for the file requirements
        st.info("#### Please ensure your file includes the following columns: :key:")
        
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            - **Username**
            - **Brand**
            - **Product**
            - **Sell Price**
            """)

        with col2:
            st.markdown("""
            - **RAM**
            - **STOCK**
            - **Source**
            """)

        uploaded_file = st.file_uploader("Choose an Excel file 📂", type=["xlsx"])

        if uploaded_file is not None:
            st.write("✅ File uploaded successfully! Processing... Please wait.")

            with st.spinner("Understanding and Fixing Your Data... :mag:"):
                df_fixed, status_counts, anomaly = process_data(uploaded_file)

            if df_fixed is not None:
                st.success("🎉 Data Fixed successfully!")

                # Display anomalous data in an expandable container
                with st.expander("📄 View Fixed vs Anomalous Data", expanded=False):
                    anomaly = anomaly.rename(columns={"Sell Price": "Original Selling Price", "Predicted_Selling_Price": "Adjusted Selling Price"})
                    st.dataframe(anomaly)
                with st.expander(":chart_with_upwards_trend: View Proportions", expanded=False):
                # Display the graph
                    st.subheader('📊 Proportion of Normal vs Anomalous Prices')
                    labels = status_counts.index
                    sizes = status_counts.values
                    plt.figure(figsize=(8, 6))
                    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#07AAE2', '#FFA500'])
                    plt.title('Proportions of Price Status', color='#07AAE2')              
                    plt.xlabel('Status', color='#07AAE2')
                    plt.ylabel('Frequency', color='#07AAE2')
                    st.pyplot(plt)
            
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
    elif choice == ":tv: Project CE":
        st.title("🔧 CE - Coming Soon")
        st.info("This feature is under development. Stay tuned for updates! 🚀")

if __name__ == "__main__":
    main()
