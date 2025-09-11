import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# ======================
# Download NLTK Resources (لو مش متسطبة عندك)
# ======================
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# ======================
# Step 1: Load Raw Data
# ======================
df = pd.read_csv("D:/IMDB reviews.project/data/raw/IMDB Dataset.csv")

# ======================
# Step 2: Preprocessing
# ======================
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)         # remove HTML tags
    text = re.sub(r"[^a-zA-Z]", " ", text)     # keep only letters
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]  # remove stopwords
    tokens = [lemmatizer.lemmatize(w) for w in tokens]   # lemmatization
    return " ".join(tokens)

df["clean_review"] = df["review"].apply(clean_text)

# ======================
# Step 3: Train/Test Split
# ======================
X = df["clean_review"].values
y = df["sentiment"].map({"positive": 1, "negative": 0}).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================
# Step 4: Tokenization & Padding
# ======================
max_words = 10000
max_len = 200

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding="post", truncating="post")
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding="post", truncating="post")

# ======================
# Step 5: Build Deep Learning Model
# ======================
model = Sequential()
model.add(Embedding(max_words, 128, input_length=max_len))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# ======================
# Step 6: Train Model
# ======================
history = model.fit(
    X_train_pad, y_train,
    validation_data=(X_test_pad, y_test),
    epochs=3,
    batch_size=64,
    verbose=1
)

# ======================
# Step 7: Evaluate
# ======================
loss, acc = model.evaluate(X_test_pad, y_test, verbose=0)
print(f"Test Accuracy: {acc*100:.2f}%")
