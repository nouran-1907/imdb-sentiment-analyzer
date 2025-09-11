import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import joblib

# ===== NLTK setup =====
nltk.data.path.append('C:/nltk_data')
for pkg in ['punkt', 'stopwords', 'wordnet']:
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg, download_dir='C:/nltk_data')

# ===== 1- Clean text =====
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)            # remove HTML
    text = re.sub(r'\d+', '', text)              # remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    return text

# ===== 2- Tokenization + Stopwords removal =====
stop_words = set(stopwords.words('english'))

def tokenize_and_remove_stopwords(text):
    tokens = word_tokenize(text)
    filtered = [word for word in tokens if word not in stop_words and len(word) > 2]
    return filtered

# ===== 3- Lemmatization =====
lemmatizer = WordNetLemmatizer()

def lemmatize_tokens(tokens):
    if not tokens:
        return []
    return [lemmatizer.lemmatize(token) for token in tokens]

# ===== 4- Combine tokens back into sentence =====
def combine_tokens(tokens):
    return " ".join(tokens)

# ===== 5- Full preprocessing pipeline =====
def preprocess_dataframe(df, text_column="review"):
    df['clean_review'] = df[text_column].apply(clean_text)
    df['tokens'] = df['clean_review'].apply(tokenize_and_remove_stopwords)
    df['lemmas'] = df['tokens'].apply(lemmatize_tokens)
    df['final_text'] = df['lemmas'].apply(combine_tokens)
    return df

# ===== 6- TF-IDF features =====
def get_tfidf_features(texts, max_features=5000, save_path=None):
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(texts)
    if save_path:
        joblib.dump(vectorizer, save_path)
    return X, vectorizer

# ===== Example run =====
if __name__ == "__main__":
    df = pd.read_csv("D:/IMDB reviews.project/data/raw/IMDB Dataset.csv")
    df = preprocess_dataframe(df, text_column="review")
    X, vectorizer = get_tfidf_features(
        df['final_text'], 
        save_path="D:/IMDB reviews.project/models/tfidf_vectorizer.pkl"
    )

    print("✅ Preprocessing Done!")
    print("Shape of TF-IDF Matrix:", X.shape)
