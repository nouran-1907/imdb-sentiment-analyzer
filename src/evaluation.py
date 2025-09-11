import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ===== Load Data =====
df = pd.read_csv("D:/IMDB reviews.project/data/raw/IMDB Dataset.csv")
df = df.sample(n=5000, random_state=42)  # sample for quick testing

# ===== TF-IDF directly on raw reviews =====
vectorizer = TfidfVectorizer(max_features=10000)
X = vectorizer.fit_transform(df["review"])
y = df["sentiment"]

# ===== Encode Labels =====
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ===== Split =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# ===== Function to evaluate =====
def evaluate_model(name, y_true, y_pred):
    print(f"\n=== {name} ===")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))

# ===== Logistic Regression =====
log_model = LogisticRegression(max_iter=200)
log_model.fit(X_train, y_train)
evaluate_model("Logistic Regression", y_test, log_model.predict(X_test))

# ===== Random Forest =====
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
evaluate_model("Random Forest", y_test, rf_model.predict(X_test))

# ===== SVM (Linear) =====
svm_model = SVC(kernel="linear")
svm_model.fit(X_train, y_train)
evaluate_model("SVM (Linear)", y_test, svm_model.predict(X_test))

# ===== Deep Learning =====
dl_model = Sequential()
dl_model.add(Dense(512, input_dim=X_train.shape[1], activation='relu'))
dl_model.add(Dropout(0.5))
dl_model.add(Dense(256, activation='relu'))
dl_model.add(Dropout(0.5))
dl_model.add(Dense(1, activation='sigmoid'))

dl_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

dl_model.fit(
    X_train.toarray(), y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

dl_pred = (dl_model.predict(X_test.toarray()) > 0.5).astype("int32")
evaluate_model("Deep Learning", y_test, dl_pred)
