<div align="center">

# 💬 Sentiment Analysis

**Classify the sentiment of tweets and text with classic ML and Transformers — then visualize it.**

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-154F5B?style=for-the-badge)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-FFD21E?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

</div>

---

## ✨ Features

- 🔍 **Live prediction** — type any text and get its sentiment instantly
- ⚖️ **Three models to choose from** — Logistic Regression, Naive Bayes and a BERT Transformer
- 📊 **Model comparison** — side-by-side metrics and confusion matrices
- 📂 **Bulk CSV analysis** — upload a CSV, pick the text column and analyze every row
- ☁️ **Visualizations** — sentiment distribution charts and word clouds

## 🧠 How it works

1. **Data** — trained on NLTK's `twitter_samples` corpus
2. **Preprocessing** — tweets are tokenized and cleaned (`src/utils.py`)
3. **Models** (`src/model.py`):
   - TF-IDF + **Logistic Regression**
   - TF-IDF + **Multinomial Naive Bayes**
   - Pre-trained **BERT** pipeline via Hugging Face Transformers
4. **Visuals** — distribution plots, confusion matrices and word clouds (`src/visualize.py`)

## 📁 Project structure

```
sentiment_analysis/
├── app.py              # Streamlit app (predict · compare · bulk CSV)
├── main.py             # CLI training & evaluation
├── src/
│   ├── model.py        # SentimentModel (LR / NB / BERT)
│   ├── utils.py        # Tweet preprocessing, NLTK setup
│   └── visualize.py    # Charts & word clouds
├── models/             # Saved LR and NB models
├── sample_data.csv     # Example CSV for bulk analysis
└── requirements.txt
```

## 🚀 Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🛠️ Tech stack

Python · scikit-learn · NLTK · Hugging Face Transformers · PyTorch · pandas · Matplotlib · Seaborn · WordCloud · Streamlit

---

<div align="center">
Made with 💜 by <a href="https://github.com/diyaverma06">Diya Verma</a> · <a href="https://www.diyaverma.in">Portfolio</a>
</div>
