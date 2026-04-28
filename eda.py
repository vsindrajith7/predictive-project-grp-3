# Basic libraries
import pandas as pd
import numpy as np
import re

# NLP
import nltk
from nltk.corpus import stopwords

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Feature Engineering
from sklearn.feature_extraction.text import TfidfVectorizer

# Save models
import pickle

nltk.download('stopwords')

df = pd.read_csv("emails.csv")

# View data
print(df.head())
print(df.info())

# Remove duplicates
df.drop_duplicates(inplace=True)

# Remove missing values
df.dropna(inplace=True)

# Check class distribution
print(df['spam'].value_counts())

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"subject:", "", text)   # remove subject tag
    text = re.sub(r"[^a-zA-Z]", " ", text) # keep only letters
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# Apply cleaning
df['clean_text'] = df['text'].apply(clean_text)

# Preview cleaned text
print(df[['text', 'clean_text']].head())

plt.figure()
df['spam'].value_counts().plot(kind='bar')
plt.title("Spam vs Not Spam Count")
plt.xlabel("Label (0 = Not Spam, 1 = Spam)")
plt.ylabel("Count")
plt.show()

df['length'] = df['text'].apply(len)

plt.figure()
sns.histplot(data=df, x='length', hue='spam', bins=50)
plt.title("Email Length Distribution")
plt.show()

spam_words = " ".join(df[df['spam'] == 1]['clean_text'])

wordcloud_spam = WordCloud(width=800, height=400).generate(spam_words)

plt.figure()
plt.imshow(wordcloud_spam)
plt.axis('off')
plt.title("Spam WordCloud")
plt.show()

ham_words = " ".join(df[df['spam'] == 0]['clean_text'])

wordcloud_ham = WordCloud(width=800, height=400).generate(ham_words)

plt.figure()
plt.imshow(wordcloud_ham)
plt.axis('off')
plt.title("Not Spam WordCloud")
plt.show()

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(df['clean_text'])
y = df['spam']

print("Feature shape:", X.shape)

