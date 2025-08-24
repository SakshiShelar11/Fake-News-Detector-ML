import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import trafilatura
import os

from sklearn.metrics import classification_report, confusion_matrix
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# ------------------- NLTK Setup -------------------
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))
ps = PorterStemmer()

def stemming(content):
    """Clean and stem text"""
    content = re.sub('[^a-zA-Z]', " ", str(content))
    content = content.lower()
    words = content.split()
    words = [ps.stem(word) for word in words if word not in STOPWORDS]
    return " ".join(words)

# ------------------- Load Data -------------------
@st.cache_data
def load_data():
    dataset_path = "WELFake_Dataset.csv"
    if os.path.exists(dataset_path):
        try:
            df = pd.read_csv(dataset_path)
        except Exception as e:
            st.error(f"❌ Error reading dataset: {e}")
            df = pd.DataFrame({
                "text": ["This is a real news article", "Breaking: Fake news spreads fast"],
                "label": [0, 1]
            })
    else:
        st.warning("⚠️ Dataset not found. Using sample data instead.")
        df = pd.DataFrame({
            "text": ["This is a real news article", "Breaking: Fake news spreads fast"],
            "label": [0, 1]
        })

    df = df.fillna(" ")
    df["content"] = df["text"].apply(stemming)

    # keep limited rows for Streamlit Cloud performance
    if len(df) > 5000:
        df = df.sample(5000, random_state=42)

    return df

# ------------------- Train Model -------------------
@st.cache_resource
def train_model(df):
    X = df["content"]
    y = df["label"]

    # Convert labels to numeric
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    vector = TfidfVectorizer(max_features=5000)
    X = vector.fit_transform(X)

    if len(np.unique(y_encoded)) > 1 and min(np.bincount(y_encoded)) >= 2:
        stratify = y_encoded
    else:
        stratify = None
        st.warning("⚠️ Not enough samples for stratified split. Proceeding without stratification.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=stratify, random_state=42
    )

    model = LogisticRegression(max_iter=500, n_jobs=-1)
    model.fit(X_train, y_train)

    return model, vector, le

# Load
df = load_data()
model, vector, le = train_model(df)

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="Fake News Detector", layout="wide")
st.title("📰 Fake News Detection with ML")

tab1, tab2, tab3 = st.tabs(["🔍 Predict News", "📊 Evaluate Model", "📁 Batch Prediction"])

# ------------------- Tab 1 -------------------
with tab1:
    st.header("Enter News Text or Paste a URL")

    input_text = st.text_area("📝 Enter News Content Here", height=200)
    url = st.text_input("🌐 Paste News URL")

    if url:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            extracted = trafilatura.extract(downloaded)
            if extracted:
                st.success("✅ Article text extracted")
                input_text = extracted
            else:
                st.error("❌ Could not extract text from this URL.")
        else:
            st.error("❌ Failed to fetch the URL.")

    if input_text:
        st.subheader("🧹 Cleaned Text Preview")
        cleaned = stemming(input_text)
        st.write(cleaned[:500] + "...")

        input_vector = vector.transform([cleaned])
        pred = model.predict(input_vector)[0]
        proba = model.predict_proba(input_vector)[0]

        st.subheader("📣 Prediction Result")
        if pred == 1:
            st.error("⚠️ The news is **Fake**.")
        else:
            st.success("✅ The news is **Real**.")

        st.info(f"🧠 Model Confidence: {round(max(proba) * 100, 2)}%")

# ------------------- Tab 2 -------------------
with tab2:
    st.header("📊 Model Performance")

    y_true = df['label']
    y_pred = model.predict(vector.transform(df['content']))

    # ✅ Use the same LabelEncoder from training
    y_true_encoded = le.transform(y_true)
    y_pred_encoded = y_pred  # model.predict already numeric

    st.subheader("📋 Classification Report")
    report = classification_report(y_true_encoded, y_pred_encoded, output_dict=True)
    st.json(report)

    st.subheader("📉 Confusion Matrix")
    cm = confusion_matrix(y_true_encoded, y_pred_encoded)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)


# ------------------- Tab 3 -------------------
with tab3:
    st.header("📁 Predict News from CSV")

    uploaded_file = st.file_uploader("Upload a CSV with a 'text' column", type=['csv'])
    if uploaded_file:
        try:
            user_df = pd.read_csv(uploaded_file)

            if 'text' not in user_df.columns:
                st.error("Uploaded CSV must contain a 'text' column.")
            else:
                user_df = user_df.fillna(" ")
                user_df['content'] = user_df['text'].apply(stemming)
                user_vector = vector.transform(user_df['content'])
                user_df['Prediction'] = model.predict(user_vector)
                user_df['Prediction'] = user_df['Prediction'].apply(lambda x: 'Fake' if x == 1 else 'Real')

                st.success("✅ Predictions completed")
                st.dataframe(user_df[['text', 'Prediction']].head(50))

                csv = user_df.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Results CSV", csv, "predicted_news.csv", "text/csv")
        except Exception as e:
            st.error(f"❌ Error processing uploaded file: {e}")






