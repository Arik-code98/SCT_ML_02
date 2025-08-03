import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


st.set_page_config(page_title="Customer Segmentation App", layout="centered")
st.title("🎯 Customer Segmentation with K-Means Clustering")

st.markdown("""
Upload your retail customer dataset to perform K-Means clustering 
and visualize customer groups based on selected behavioral features.
""")


uploaded_file = st.file_uploader("📤 Upload CSV File", type=["csv"])


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    st.write("### 📄 Preview of Dataset")
    st.dataframe(df.head())

    
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    st.write("### 🔢 Select Features for Clustering")
    selected_features = st.multiselect(
        "Choose two numeric features for clustering",
        options=numeric_cols,
        default=["Annual Income (k$)", "Spending Score (1-100)"] if "Annual Income (k$)" in numeric_cols else numeric_cols[:2]
    )

    if len(selected_features) != 2:
        st.warning("⚠️ Please select exactly two numeric features.")
    else:
        X = df[selected_features]

        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

       
        st.write("### ⚙️ K-Means Configuration")
        k = st.slider("Select number of clusters (K)", 2, 10, 5)

        
        kmeans = KMeans(n_clusters=k, random_state=42)
        df['Cluster'] = kmeans.fit_predict(X_scaled)

        
        st.write("### 📊 Data with Cluster Labels")
        st.dataframe(df[[*selected_features, 'Cluster']].head())

        
        st.write("### 🖼️ Cluster Visualization")
        fig, ax = plt.subplots()
        sns.scatterplot(
            x=selected_features[0], y=selected_features[1],
            hue='Cluster', data=df, palette='viridis', ax=ax
        )
        ax.set_title("Customer Segmentation")
        ax.set_xlabel(selected_features[0])
        ax.set_ylabel(selected_features[1])
        st.pyplot(fig)

        
        st.write("### 📌 Cluster Sizes")
        cluster_counts = df['Cluster'].value_counts().sort_index()
        st.bar_chart(cluster_counts)

else:
    st.info("📁 Please upload a CSV file to get started.")
