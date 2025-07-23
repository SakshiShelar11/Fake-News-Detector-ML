# 📰 Fake News Detection with Machine Learning

A web-based application built using **Streamlit** and **Machine Learning** that detects whether a given news article is **real or fake** based on its content. It also supports URL-based prediction, model evaluation, and batch predictions via CSV upload.

---

## 🚀 Features

- 🔍 Predict if a news article is *Real* or *Fake* using ML
- 🌐 Paste a news **URL** to auto-extract and analyze the article
- 📊 Visualize model **performance** with classification reports and confusion matrix
- 📁 Upload a **CSV file** for bulk predictions
- 💡 Shows **prediction confidence** percentage
- 🧹 Text preprocessing using stemming and stopword removal

---

## 🧠 Technologies Used

- Python 3.x
- Streamlit
- Scikit-learn
- Pandas, NumPy
- NLTK
- Matplotlib, Seaborn
- Trafilatura *(for extracting article text from URLs)*

---

## 📁 Dataset Information

This project uses the **WELFake Dataset**, a large-scale labeled dataset for fake news detection.

📌 **Dataset Name**: `WELFake_Dataset.csv`  
📌 **Columns**: `title`, `text`, `label`  
📌 **Target**:  
- `1` → Fake  
- `0` → Real  

### 🔗 Download Dataset

Since the dataset is not included in this GitHub repo (due to file size constraints), you can download it manually:

👉 [Download WELFake Dataset from Kaggle](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification?select=WELFake_Dataset.csv)

> **After downloading**, place `WELFake_Dataset.csv` in the project root directory.

---
