import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.ensemble import GradientBoostingClassifier
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Load the trained model
with open('gbc.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
    
    # Load the TF-IDF vectorizer
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)


# Text preprocessing function
def preprocess_text(text):
    words = word_tokenize(text.lower())  # Tokenization and lowercase
    words = [stemmer.stem(word) for word in words if word.isalnum()]  # Stemming and removing non-alphanumeric
    words = [word for word in words if word not in stop_words]  # Removing stopwords
    return ' '.join(words)

# Streamlit app
def main():
    st.title('Email Spam Detection')

    input_email = st.text_area('Enter the email text here:')
    if st.button('Predict'):
        if input_email:
            preprocessed_email = preprocess_text(input_email)
            prediction = model.predict([preprocessed_email])
            if prediction[0] == 1:
                st.error('This email is classified as spam.')
            else:
                st.success('This email is not spam.')

if __name__ == '__main__':
    main()
