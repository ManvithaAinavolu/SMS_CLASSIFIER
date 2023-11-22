import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
#load the vectorizer file
# Load the vectorizer file
try:
    with open('vectorizer.pk1', 'rb') as file:
        tfdif = pickle.load(file)
except FileNotFoundError:
    st.error("Error: 'vectorizer.pk1' file not found.")
except Exception as e:
    st.error(f"Error loading 'vectorizer.pkl': {e}")

# Load the model file
try:
    with open('model.pk1', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Error: 'model.pk1' file not found.")
except Exception as e:
    st.error(f"Error loading 'model.pk1': {e}")

# Re-save the model with the current scikit-learn version
with open('vectorizer.pk1', 'wb') as file:
    pickle.dump(tfdif, file)

with open('model.pk1', 'wb') as file:
    pickle.dump(model, file)


import string
import nltk
import os

nltk.data.path.append("C:\\Users\\DELL\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\nltk")
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# Download the 'stopwords' resource
try:
    nltk.download('stopwords')
except Exception as e:
    st.error(f"Error downloading 'stopwords': {e}")

# Download the 'punkt' resource
try:
    nltk.download('punkt')
except Exception as e:
    st.error(f"Error downloading 'punkt': {e}")

ps=PorterStemmer()

def trans_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    text=[word for word in text if word.isalnum()]
    text=[word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    text=[ps.stem(word) for word in text]
    return " ".join(text)
#saving stramlit code

st.title('Email Spam Classifier')
input_sms=st.text_area('Enter Message')

if st.button('Predict'):
    #preprocess
    transformed_sms=trans_text(input_sms)
    #vectorize
    vector_input=tfdif.transform([transformed_sms])
    #result
    result=model.predict(vector_input)[0]
    #display
    if result==1:
        st.header('Spam')
    else:
        st.header('Not Spam')