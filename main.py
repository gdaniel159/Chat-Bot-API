from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import load_model
import json
import random

intents = json.loads(open('intents.json', 'r', encoding='utf-8').read())

# Load the vectorizer vocabulary
vectorizer_vocab = pickle.load(open('vectorizer_vocab.pkl', 'rb'))

# Reemplaza el tokenizador por defecto por tu propia función de tokenización
def custom_tokenizer(text):
    words_list = nltk.word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word.lower()) for word in words_list]
    return lemmatized_words

# Crea una instancia de CountVectorizer con tu función de tokenización
vectorizer = CountVectorizer(lowercase=True, tokenizer=custom_tokenizer, vocabulary=vectorizer_vocab)

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app)

lemmatizer = WordNetLemmatizer()

# Carga las palabras y las clases preprocesadas previamente
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

model = load_model("chatbot_model.h5")

@app.route('/api', methods=['POST'])
def handle_api_request():

    data = request.get_json()
    user_message = data.get('user_message')

    print("User Message:", user_message) 

    # Aqui mandaremos la response

    # Genera la respuesta utilizando el modelo
    response = generate_response(user_message)

    print("Generated Response:", response)

    response_data = {
        'response' : response
    }

    return jsonify(response_data), 200, {'Content-Type': 'application/json; charset=utf-8'}

def generate_response(user_message):

    user_message_bag = preprocess_message(user_message, vectorizer)
    predicted_probabilities = model.predict(np.array([user_message_bag]))
    predicted_class = np.argmax(predicted_probabilities)
    response = get_response_by_class(classes[predicted_class])
    return response

def preprocess_message(message, vectorizer):

    words_list = nltk.word_tokenize(message)
    print("Tokenized words:", words_list)  # Add this line to check tokenization
    words_list = list(set([lemmatizer.lemmatize(word.lower()) for word in words_list]))

    bag = [0] * len(vectorizer.get_feature_names_out())
    for word in words_list:
        if word in vectorizer.vocabulary_:
            bag[vectorizer.vocabulary_[word]] = 1

    return np.array(bag)

def get_response_by_class(tag):

    for intent in intents['intents']:
        if intent['tag'] == tag:
            return np.random.choice(intent['responses'])
    
    print("No se encontró una respuesta para la etiqueta.")
    return "Lo siento, no entiendo lo que dices."

if __name__ == '__main__':
    app.run(debug=True)