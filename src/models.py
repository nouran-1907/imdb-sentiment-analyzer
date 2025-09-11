import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing import clean_text, tokenize_and_remove_stopwords, lemmatize_tokens

# ===== 1- Load dataset (sample only) =====
df = pd.read_csv("D:/IMDB reviews.project/data/raw/IMDB Dataset.csv")

# take a random sample (مثلاً 5000 ريفيو بدل 50,000)
df = df.sample(5000, random_state=42).reset_index(drop=True)

# ===== 2- Apply preprocessing (from saved functions) =====
df['clean_review'] = df['review'].apply(clean_text)
df['tokens'] = df['clean_review'].apply(tokenize_and_remove_stopwords)
df['lemmas'] = df['tokens'].apply(lemmatize_tokens)
df['final_text'] = df['lemmas'].apply(lambda x: " ".join(x))

# ===== 3- TF-IDF =====
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['final_text'])
y = df['sentiment']

# ===== 4- Train/test split =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===== 5- Logistic Regression =====
log_model = LogisticRegression(max_iter=200)
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)
print("✅ Logistic Regression Accuracy:", accuracy_score(y_test, log_pred))

# ===== 6- Random Forest =====
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print("✅ Random Forest Accuracy:", accuracy_score(y_test, rf_pred))

# ===== 7- SVM =====
svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
print("✅ SVM Accuracy:", accuracy_score(y_test, svm_pred))
