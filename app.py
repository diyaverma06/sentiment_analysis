import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model import SentimentModel
from src.utils import download_nltk_data, process_tweet
from src.visualize import plot_sentiment_distribution, plot_confusion_matrix, plot_word_cloud
from nltk.corpus import twitter_samples
from sklearn.model_selection import train_test_split

# Page Config
st.set_page_config(page_title="Sentiment Analyzer", layout="wide")

@st.cache_resource
def load_data_and_train():
    """
    This function handles the heavy lifting of loading the NLTK data and training our models.
    """
    download_nltk_data()
    
    pos_tweets = twitter_samples.strings('positive_tweets.json')
    neg_tweets = twitter_samples.strings('negative_tweets.json')
    
    y_pos = [1] * len(pos_tweets)
    y_neg = [0] * len(neg_tweets)
    
    X = pos_tweets + neg_tweets
    y = y_pos + y_neg
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 1. Logistic Regression
    model_lr = SentimentModel(model_type='logistic')
    model_path_lr = 'models/sentiment_model_lr.pkl'
    if os.path.exists(model_path_lr):
        model_lr.load_model(model_path_lr)
    else:
        model_lr.train(X_train, y_train)
        model_lr.save_model(model_path_lr)

    # 2. Naive Bayes
    model_nb = SentimentModel(model_type='naive_bayes')
    model_path_nb = 'models/sentiment_model_nb.pkl'
    if os.path.exists(model_path_nb):
        model_nb.load_model(model_path_nb)
    else:
        model_nb.train(X_train, y_train)
        model_nb.save_model(model_path_nb)

    # 3. BERT (Transformer)
    model_bert = SentimentModel(model_type='bert')
    # No training needed for BERT, just loading.
    
    return model_lr, model_nb, model_bert, X_test, y_test, X_train, y_train

# Load models and data
with st.spinner("Loading models (including BERT ~400MB usually cached)..."):
    model_lr, model_nb, model_bert, X_test, y_test, X_train, y_train = load_data_and_train()

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Model Comparison", "Bulk Analysis", "Visualizations", "Project Info"])

if page == "Home":
    st.title("😊/☹️ Sentiment Analysis App")
    st.write("Enter a tweet or sentence below to analyze its sentiment.")
    
    model_choice = st.radio("Choose Model:", ["Logistic Regression", "Naive Bayes", "Transformer (BERT)"])
    if model_choice == "Logistic Regression":
        active_model = model_lr
    elif model_choice == "Naive Bayes":
        active_model = model_nb
    else:
        active_model = model_bert
    
    user_input = st.text_area("Enter text here:", height=100)
    show_cleaned = st.checkbox("Show cleaned/processed text")
    
    if st.button("Analyze"):
        if user_input:
            with st.spinner("Analyzing..."):
                # Prediction
                prediction = active_model.predict([user_input])[0]
                proba = active_model.predict_proba([user_input])[0]
                
                sentiment = "Positive" if prediction == 1 else "Negative"
                confidence = proba[prediction]
                
                # Display results vividly
                st.subheader(f"Sentiment: {sentiment}")
                st.metric("Confidence Score", f"{confidence:.2%}")
                
                if show_cleaned:
                    processed_tokens = process_tweet(user_input)
                    st.text("Processed Tokens:")
                    st.info(processed_tokens)
                
                if sentiment == "Positive":
                    st.success("Analysis Complete!")
                else:
                    st.error("Analysis Complete!")
        else:
            st.warning("Please enter some text.")

elif page == "Model Comparison":
    st.title("📊 Model Comparison")
    st.write("Evaluating models on test data...")
    st.info("Note: BERT evaluation might take significantly longer due to CPU computation.")
    
    if st.button("Run Evaluation"):
        with st.spinner("Running evaluation... (this may take a minute)"):
            acc_lr, report_lr, conf_lr, _ = model_lr.evaluate(X_test, y_test)
            acc_nb, report_nb, conf_nb, _ = model_nb.evaluate(X_test, y_test)
            
            # For BERT, let's just evaluate on a small subset to save time if running on CPU
            subset_size = 100
            st.warning(f"Evaluating BERT on random {subset_size} samples to save time.")
            idx = np.random.choice(len(X_test), subset_size, replace=False)
            X_bert = [X_test[i] for i in idx]
            y_bert = [y_test[i] for i in idx]
            acc_bert, report_bert, conf_bert, _ = model_bert.evaluate(X_bert, y_bert)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Logistic Regression")
                st.metric("Accuracy", f"{acc_lr:.4f}")
                fig_lr = plot_confusion_matrix(conf_lr, classes=['Neg', 'Pos'], title='LR Confusion Matrix')
                st.pyplot(fig_lr)

            with col2:
                st.subheader("Naive Bayes")
                st.metric("Accuracy", f"{acc_nb:.4f}")
                fig_nb = plot_confusion_matrix(conf_nb, classes=['Neg', 'Pos'], title='NB Confusion Matrix')
                st.pyplot(fig_nb)
                
            with col3:
                st.subheader("BERT")
                st.metric("Accuracy", f"{acc_bert:.4f}")
                fig_bert = plot_confusion_matrix(conf_bert, classes=['Neg', 'Pos'], title='BERT Confusion Matrix')
                st.pyplot(fig_bert)

elif page == "Bulk Analysis":
    st.title("📂 Bulk CSV Analysis")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            
            text_col = st.selectbox("Select the column containing text:", df.columns)
            
            # Model selection for bulk
            bulk_model_choice = st.selectbox("Select Model for Bulk Analysis:", ["Logistic Regression", "BERT (Slower but Accurate)"])
            
            if st.button("Run Analysis"):
                if bulk_model_choice == "Logistic Regression":
                    model = model_lr
                else:
                    model = model_bert
                
                with st.spinner("Analyzing..."):
                    text_data = df[text_col].astype(str)
                    
                    if bulk_model_choice == "BERT (Slower but Accurate)":
                         # Convert Series to list for pipeline
                        predictions = model.predict(text_data.tolist())
                        probas = model.predict_proba(text_data.tolist())
                    else:
                        predictions = model.predict(text_data)
                        probas = model.predict_proba(text_data)
                        
                    confidence = [max(p) for p in probas]
                    
                    df['Predicted_Sentiment'] = ["Positive" if p == 1 else "Negative" for p in predictions]
                    df['Confidence'] = confidence
                
                st.success("Analysis Complete!")
                st.dataframe(df)
                
                # Download
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Results as CSV",
                    csv,
                    "sentiment_results.csv",
                    "text/csv",
                    key='download-csv'
                )
        except Exception as e:
            st.error(f"Error processing file: {e}")

elif page == "Visualizations":
    st.title("🎨 Visualizations")
    st.write("Word clouds generated from the training dataset.")
    
    pos_text = [t for t, label in zip(X_train, y_train) if label == 1]
    neg_text = [t for t, label in zip(X_train, y_train) if label == 0]
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Positive Tweets")
        with st.spinner("Generating Positive Word Cloud..."):
            fig_pos = plot_word_cloud(pos_text, title="Positive Word Cloud")
            st.pyplot(fig_pos)
            
    with col2:
        st.subheader("Negative Tweets")
        with st.spinner("Generating Negative Word Cloud..."):
            fig_neg = plot_word_cloud(neg_text, title="Negative Word Cloud")
            st.pyplot(fig_neg)

elif page == "Project Info":
    st.title("ℹ️ Project Info")
    st.markdown("""
    **Sentiment Analysis Project**
    
    **Features:**
    - **Models**: Logistic Regression, Naive Bayes, **BERT (DistilBERT)**.
    - **Features**: Real-time analysis, Bulk CSV, Word Clouds.
    
    **About BERT:**
    We used `distilbert-base-uncased-finetuned-sst-2-english` from Hugging Face. 
    It understands context! 
    Try inputs like *"Are you deaf?"* -> BERT knows this is likely Negative/Insult, while simple models might be confused.
    """)
