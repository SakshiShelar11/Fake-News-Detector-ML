import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from newspaper import Article
import trafilatura

# Download NLTK stopwords if not already downloaded
import nltk
nltk.download('stopwords')

# Preprocessing
ps = PorterStemmer()

def stemming(content):
    content = re.sub('[^a-zA-Z]', " ", content)
    content = content.lower()
    words = content.split()
    words = [ps.stem(word) for word in words if word not in stopwords.words('english')]
    return " ".join(words)

# Cache data loading
@st.cache_data
def load_data():
    df = pd.read_csv('WELFake_Dataset.csv')
    df = df.fillna(' ')
    df['content'] = df['text']
    df = df.head(500)  # reduce for faster testing
    df['content'] = df['content'].apply(stemming)
    return df

# Cache model training
@st.cache_resource
def train_model(df):
    X = df['content'].values
    y = df['label'].values
    vector = TfidfVectorizer()
    X = vector.fit_transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    return model, vector, X_test, Y_test

# Load data and model
df = load_data()
model, vector, X_test, Y_test = train_model(df)

# ----------- UI ------------
st.set_page_config(page_title="Fake News Detector", layout="wide")
st.title("📰 Fake News Detection with ML")

# --- Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Predict News", "📊 Evaluate Model", "📁 Batch Prediction"])

# --- Tab 1: Single News Prediction ---
with tab1:
    st.header("Enter News Manually or Paste URL")

    input_text = st.text_area("📝 Enter News Content Here", height=200)

    url = st.text_input("🌐 Paste News URL")
if url:
    downloaded = trafilatura.fetch_url(url)
    if downloaded:
        input_text = trafilatura.extract(downloaded)
        st.success("✅ Article text extracted.")
        st.text_area("Extracted Article", input_text, height=200)
    else:
        st.error("❌ Failed to extract article. Try a different site.")

    if input_text:
        st.subheader("🧹 Cleaned Text Preview")
        cleaned = stemming(input_text)
        st.write(cleaned)

        input_vector = vector.transform([cleaned])
        pred = model.predict(input_vector)[0]
        proba = model.predict_proba(input_vector)[0]

        st.subheader("📣 Prediction Result")
        if pred == 1:
            st.error("⚠️ The news is **Fake**.")
        else:
            st.success("✅ The news is **Real**.")

        st.info(f"🧠 Model Confidence: {round(max(proba) * 100, 2)}%")

# --- Tab 2: Model Evaluation ---
with tab2:
    st.header("📊 Model Performance")

    y_true = df['label']
    y_pred = model.predict(vector.transform(df['content']))

    st.subheader("📋 Classification Report")
    report = classification_report(y_true, y_pred, output_dict=True)
    st.json(report)

    st.subheader("📉 Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# --- Tab 3: Batch Prediction via CSV Upload ---
with tab3:
    st.header("📁 Predict News from CSV")

    uploaded_file = st.file_uploader("Upload a CSV file with a 'text' column", type=['csv'])
    if uploaded_file is not None:
        user_df = pd.read_csv(uploaded_file)

        if 'text' not in user_df.columns:
            st.error("Uploaded CSV must contain a 'text' column.")
        else:
            user_df = user_df.fillna(' ')
            user_df['content'] = user_df['text'].apply(stemming)
            user_vector = vector.transform(user_df['content'])
            user_df['Prediction'] = model.predict(user_vector)
            user_df['Prediction'] = user_df['Prediction'].apply(lambda x: 'Fake' if x == 1 else 'Real')
            st.success("✅ Predictions completed.")
            st.dataframe(user_df[['text', 'Prediction']])

            csv = user_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Results CSV", csv, "predicted_news.csv", "text/csv")

