import streamlit as st
import joblib
from src.preprocessing import clean_text, tokenize_and_remove_stopwords, lemmatize_tokens
from src.summarization import summarize_review

# ===== Streamlit App =====
st.title("Movie Reviews Sentiment & Summarizer 🎬")

# Text input
user_input = st.text_area("Write a movie review:")

if st.button("Analyze Review"):
    if not user_input.strip():
        st.warning("Please enter a review!")
    else:
        # ===== Preprocessing =====
        clean = clean_text(user_input)
        tokens = tokenize_and_remove_stopwords(clean)
        lemmas = lemmatize_tokens(tokens)
        final_text = " ".join(lemmas)
        
        # ===== Load vectorizer & model =====
        vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
        model = joblib.load("models/logistic_model.joblib")
        
        # ===== Predict sentiment =====
        X_input = vectorizer.transform([final_text])
        pred = model.predict(X_input)[0]
        sentiment_label = "Positive" if pred == 1 else "Negative"
        st.success(sentiment_label)

        st.success(pred)
        
        # ===== Summarize review =====
        summary = summarize_review(user_input, num_sentences=2)
        st.subheader("Summary:")
        if summary.strip():
            st.write(summary)
        else:
            st.info("Couldn't generate a summary for this input.")
