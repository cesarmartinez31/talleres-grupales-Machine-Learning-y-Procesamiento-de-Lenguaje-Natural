#!/usr/bin/python
import pandas as pd
import joblib
import spacy
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Cargar el modelo y el vectorizador
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
nlp = spacy.load('en_core_web_sm')

# Lista de géneros en el mismo orden que fueron usados para entrenar el modelo
GENRE_LABELS = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
                'Fantasy', 'Film-Noir', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Romance',
                'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western']

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    lemmatized_tokens = []
    for token in nlp(text):
        if token.pos_ == "VERB":
            lemmatized_tokens.append(token.lemma_)
        elif token.text not in stopwords.words('english') and not token.text.isdigit():
            lemmatized_tokens.append(token.text)
    clean_text = ' '.join(lemmatized_tokens)
    return clean_text

def predict_genres(plot):
    clean_plot = preprocess_text(plot)
    input_data_encoded = vectorizer.transform([clean_plot])
    predicted_genres_encoded = model.predict(input_data_encoded)
    predicted_genres = [GENRE_LABELS[i] for i, value in enumerate(predicted_genres_encoded[0]) if value == 1]
    return predicted_genres
