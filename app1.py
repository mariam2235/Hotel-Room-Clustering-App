import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    with open("hotel_clustering_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

iso     = model["iso"]
scaler  = model["scaler"]
kmeans  = model["kmeans"]
pca     = model["pca"]
features = model["features"]

# ---------------- Page Title ----------------
st.title("🏨 Hotel Room Clustering App")
st.write("This app uses the trained KMeans clustering model to categorize hotel rooms.")

# ---------------- Upload CSV ----------------
uploaded_file = st.file_uploader("Upload Rooms CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data")
    st.dataframe(df.head())

    # Check if required features exist
    missing = [f for f in features if f not in df.columns]
    if missing:
        st.error(f"Missing required feature columns: {missing}")
    else:
        # Scale data
        X_scaled = scaler.transform(df[features])

        # Predict clusters
        cluster_labels = kmeans.predict(X_scaled)
        df["cluster"] = cluster_labels

        st.write("### Clustered Data")
        st.dataframe(df)

        # Optionally apply PCA for visualization
        X_pca = pca.transform(X_scaled)
        df["PCA1"] = X_pca[:,0]
        df["PCA2"] = X_pca[:,1]

        st.write("### PCA Scatter Plot")
        st.scatter_chart(df, x="PCA1", y="PCA2", color="cluster")

        # Download clustered file
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Clustered CSV",
            data=csv,
            file_name="clustered_rooms.csv",
            mime="text/csv"
        )

# ---------------- Single Prediction Form ----------------
st.write("## Predict cluster for a single room")

with st.form("form"):
    price = st.number_input("Price", min_value=0.0)
    revenue = st.number_input("Total Revenue", min_value=0.0)
    booking_count = st.number_input("Booking Count", min_value=0)
    avg_stay = st.number_input("Average Stay Days", min_value=0.0)
    is_premium = st.selectbox("Is Premium Room?", [0,1])

    submit = st.form_submit_button("Predict")

if submit:
    # Compute engineered features
    log_price = np.log1p(price)
    log_revenue = np.log1p(revenue)
    log_bookings = np.log1p(booking_count)
    log_avg_stay = np.log1p(avg_stay)
    rev_per_day = revenue / (avg_stay + 1)
    log_rev_per_day = np.log1p(rev_per_day)

    row = pd.DataFrame([[
        log_price, log_revenue, log_bookings,
        log_avg_stay, log_rev_per_day, is_premium
    ]], columns=features)

    scaled = scaler.transform(row)
    cluster_pred = kmeans.predict(scaled)[0]

    st.success(f"Predicted Cluster: {cluster_pred}")


