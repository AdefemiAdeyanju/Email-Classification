import streamlit as st
import nltk
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
data = pd.read_csv('mail_spam.csv') 

# Preprocessing
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    words = word_tokenize(text.lower())  # Tokenization and lowercase
    words = [stemmer.stem(word) for word in words if word.isalnum()]  # Stemming and removing non-alphanumeric
    words = [word for word in words if word not in stop_words]  # Removing stopwords
    return ' '.join(words)

data['processed_text'] = data['text'].apply(preprocess_text)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Extraction
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Model Training with XGBoost
gbc = GradientBoostingClassifier()
gbc.fit(X_train_vectorized, y_train)

# Streamlit App
st.title("Email Spam Detection")

user_input = st.text_area("Enter a message:", "Your message here...")

if st.button("Predict"):
    processed_input = preprocess_text(user_input)
    input_vectorized = vectorizer.transform([processed_input]).toarray()
    prediction = gbc.predict(input_vectorized)
    result = "Spam" if prediction[0] == 1 else "Ham"
    st.write(f"Prediction: {result}")
