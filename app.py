import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from bs4 import BeautifulSoup

st.set_page_config(page_title="Fake News Detector", layout="wide")

st.title("📰 Fake News Detection using Machine Learning")

# -------------------------------
# Load dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("news.csv")  # Ensure file exists
    return df

df = load_data()

# -------------------------------
# Train model
# -------------------------------
@st.cache_resource
def train_model(df):
    le = LabelEncoder()
    y = le.fit_transform(df["label"].values)

    # Convert y → numpy int array explicitly
    y = np.array(y, dtype=int)

    # Safety check: at least 2 samples per class
    if len(np.unique(y)) > 1 and min(np.bincount(y)) >= 2:
        X_train, X_test, y_train, y_test = train_test_split(
            df["text"].astype(str), y, test_size=0.2, random_state=42
        )

        vector = TfidfVectorizer(stop_words="english", max_features=5000)
        X_train_vec = vector.fit_transform(X_train)
        X_test_vec = vector.transform(X_test)

        model = LogisticRegression(max_iter=500)
        model.fit(X_train_vec, y_train)

        y_pred = model.predict(X_test_vec)

        # Classification report with string class names
        report = classification_report(
            y_test,
            y_pred,
            target_names=list(le.classes_.astype(str)),
            output_dict=True
        )

        return model, vector, le, report
    else:
        return None, None, None, None


model, vector, le, report = train_model(df)

# -------------------------------
# Tabs for navigation
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dataset",
    "🤖 Model Performance",
    "🔎 Predict Manually",
    "🌐 Predict via URL / File"
])

# -------------------------------
# Tab 1: Dataset
# -------------------------------
with tab1:
    st.write("### Dataset Preview")
    st.dataframe(df.head())
    st.write(f"Dataset Shape: {df.shape}")

# -------------------------------
# Tab 2: Model Performance
# -------------------------------
with tab2:
    if model is not None:
        st.success("✅ Model trained successfully!")

        st.write("### Classification Report")
        st.json(report)

        # Confusion Matrix
        st.write("### Confusion Matrix")
        y_true = le.transform(df["label"].values)
        y_pred = model.predict(vector.transform(df["text"].astype(str)))

        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=le.classes_,
                    yticklabels=le.classes_,
                    ax=ax)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)
    else:
        st.error("❌ Model training failed. Please check dataset.")

# -------------------------------
# Tab 3: Manual Prediction
# -------------------------------
with tab3:
    if model is not None:
        st.write("### Enter News Text")
        user_input = st.text_area("Enter news text to check if it's Fake or Real:")

        if st.button("Predict", key="manual"):
            if user_input.strip() != "":
                vec_input = vector.transform([user_input])
                prediction = model.predict(vec_input)[0]
                pred_label = le.inverse_transform([prediction])[0]

                st.write(f"**Prediction:** {pred_label}")
            else:
                st.warning("Please enter some text before predicting.")
    else:
        st.error("❌ Model not available.")

# -------------------------------
# Tab 4: URL / File Prediction
# -------------------------------
with tab4:
    if model is not None:
        st.write("### Predict from a URL")
        url_input = st.text_input("Enter a news article URL:")

        if st.button("Predict from URL"):
            if url_input.strip():
                try:
                    response = requests.get(url_input, timeout=5)
                    soup = BeautifulSoup(response.text, "html.parser")
                    paragraphs = " ".join([p.get_text() for p in soup.find_all("p")])
                    if paragraphs:
                        vec_input = vector.transform([paragraphs])
                        prediction = model.predict(vec_input)[0]
                        pred_label = le.inverse_transform([prediction])[0]
                        st.write(f"**Prediction for URL:** {pred_label}")
                    else:
                        st.warning("No text content found on this page.")
                except Exception as e:
                    st.error(f"Error fetching URL: {e}")

        st.write("---")
        st.write("### Predict from a File")
        uploaded_file = st.file_uploader("Upload a CSV file with a 'text' column", type=["csv"])

        if uploaded_file is not None:
            try:
                file_df = pd.read_csv(uploaded_file)
                if "text" in file_df.columns:
                    vec_input = vector.transform(file_df["text"].astype(str))
                    predictions = model.predict(vec_input)
                    file_df["prediction"] = le.inverse_transform(predictions)
                    st.write("### Predictions on Uploaded File")
                    st.dataframe(file_df)
                else:
                    st.error("Uploaded CSV must contain a 'text' column.")
            except Exception as e:
                st.error(f"Error processing file: {e}")
    else:
        st.error("❌ Model not available.")
