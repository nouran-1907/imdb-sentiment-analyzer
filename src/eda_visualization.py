import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# 1. Load Data
df = pd.read_csv("data/raw/IMDB Dataset.csv")



# Show basic info
print(df.head())
print(df.info())
print(df['sentiment'].value_counts())

# 2. Plot distribution of sentiments
plt.figure(figsize=(6,4))
sns.countplot(x="sentiment", data=df, hue="sentiment", palette="viridis", legend=False)
plt.title("Distribution of Sentiments")
plt.show()

# 3. WordCloud for Positive reviews
positive_text = " ".join(df[df['sentiment']=='positive']['review'].astype(str))
wordcloud_pos = WordCloud(width=800, height=400, background_color="white").generate(positive_text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud_pos, interpolation="bilinear")
plt.axis("off")
plt.title("WordCloud - Positive Reviews")
plt.show()

# 4. WordCloud for Negative reviews
negative_text = " ".join(df[df['sentiment']=='negative']['review'].astype(str))
wordcloud_neg = WordCloud(width=800, height=400, background_color="black").generate(negative_text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud_neg, interpolation="bilinear")
plt.axis("off")
plt.title("WordCloud - Negative Reviews")
plt.show()

