import random
import json
import pickle
import numpy as np

import nltk
nltk.download('punkt_tab')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

import tensorflow as tf

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

# load pickles and model
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = tf.keras.models.load_model('chatbot_model.h5')

# we can use the train model but it returns numerical data
# so we need to convert it to text - 4 methods for this task

# tokenizes the input sentence and lemmatizes each token to its root form
def clean_up_sentence(sentence):
    sentence_tokens = nltk.word_tokenize(sentence)
    clean_sentence_tokens = [lemmatizer.lemmatize(word) for word in sentence_tokens]
    return clean_sentence_tokens

# convert sentence to a bag of words (1s and 0s) 
def bag_of_words(sentence):
    sentence_tokens = clean_up_sentence(sentence)
    bag = [0] * len(words) # by default 0
    for token in sentence_tokens:
        for i, word in enumerate(words):
            if token == word:
                bag[i] = 1

    return np.array(bag)

# using model to predict the intent of the input sentence.
def predict_class(sentence):
    bow = bag_of_words(sentence)
    results = model.predict(np.array([bow]))[0] # this gives predicted intent probabilities

    ERROR_THRESHOLD = 0.25
    filtered_results = [[intent, probability] for intent, probability in enumerate(results) if probability > ERROR_THRESHOLD]
    filtered_results.sort(key=lambda x: x[1], reverse=True) # highest probability first
    
    sorted_results = []
    for res in filtered_results:
        sorted_results.append({'intent': classes[res[0]], 'probability': str(res[1])})

    return sorted_results

# fetch a random response corresponding to the predicted intent
def get_response(predicted_intents_list, intents_json):
    tag = predicted_intents_list[0]['intent'] # get first element = intent prediction with highest probability
    list_of_intents = intents_json['intents'] # get the entire JSON file

    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# provide an interactive loop for the chatbot
while True:
    user_message = input('')
    predicted_intents_list = predict_class(user_message)
    bot_message = get_response(predicted_intents_list, intents)
    print(bot_message)
