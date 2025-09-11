# src/train_models.py
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from preprocessing import clean_text, tokenize_and_remove_stopwords, lemmatize_tokens

# ===== 0- Create folders =====
if not os.path.exists("models"):
    os.makedirs("models")

# ===== 1- Load data =====
df = pd.read_csv("data/raw/IMDB Dataset.csv")

# ===== 2- Sampling for faster training =====
df = df.sample(n=5000, random_state=42)

# ===== 3- Preprocessing =====
df['clean_review'] = df['review'].apply(clean_text)
df['tokens'] = df['clean_review'].apply(tokenize_and_remove_stopwords)
df['lemmas'] = df['tokens'].apply(lemmatize_tokens)
df['final_text'] = df['lemmas'].apply(lambda x: " ".join(x))

# ===== 4- TF-IDF =====
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['final_text'])
y = df['sentiment']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ===== 5- Split data =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# ===== 6- Logistic Regression =====
log_model = LogisticRegression(max_iter=200)
log_model.fit(X_train, y_train)
joblib.dump(log_model, "models/logistic_model.joblib")

# ===== 7- Random Forest =====
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, "models/random_forest_model.joblib")

# ===== 8- Linear SVM (faster) =====
svm_model = LinearSVC()
svm_model.fit(X_train, y_train)
joblib.dump(svm_model, "models/svm_model.joblib")

# ===== 9- Deep Learning =====
dl_model = Sequential([
    Dense(512, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
dl_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
dl_model.fit(
    X_train.toarray(), y_train,
    validation_split=0.2,
    epochs=3,              # أقل
    batch_size=256,        # أكبر
    callbacks=[early_stop],
    verbose=1
)
dl_model.save("models/deep_learning_model.h5")

# ===== 10- Save vectorizer & label encoder =====
joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")
joblib.dump(le, "models/label_encoder.joblib")

print("✅ All models and vectorizer saved successfully (fast version)!")
