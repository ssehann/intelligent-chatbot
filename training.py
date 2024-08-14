import random
import json
import pickle
import numpy as np

import nltk
nltk.download('punkt_tab')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
# recognize all forms of verb i.e. work, working, worked, works is all the same

import tensorflow as tf
# from tensorflow.python.keras import Sequential
# from tensorflow.python.keras.layers import Dense, Activation, Dropout


lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read()) # json object -> python dict

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# extract data from 
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag'])) # associate a list of tokens with a category

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# print(documents)
    # [(['Hi'], 'greeting'), 
    # (['How', 'are', 'you'], 'greeting'), 
    # (['Is', 'anyone', 'there', '?'], 'greeting'), 
    # (['Hello'], 'greeting'), 
    # (['Good', 'day'], 'greeting'), 
    # (['Whats', 'up'], 'greeting') ...]

# lemmatize = reduce words to their root form 
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words)) # remove duplicates
classes = sorted(set(classes))

# serialize Python object into a pkl file(byte stream) using wb (write binary) mode
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# actual machine learning part
# first create training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    # create the bag of words array with consistent length
    bag = [1 if word in word_patterns else 0 for word in words]
    
    # check if bag has the right shape
    if len(bag) != len(words):
        print("Inconsistent bag length detected")
    
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    # check if output_row has the right shape
    if len(output_row) != len(classes):
        print("Inconsistent output_row length detected")
    
    training.append([bag, output_row])

# final processing before neural network
random.shuffle(training)
training = np.array(training, dtype=object)  # dtype=object to prevent issues with shape
train_x = np.array([item[0] for item in training])
train_y = np.array([item[1] for item in training])

# building neural network
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(train_y[0]), activation='softmax'))
#stochastic gradient descent for weight update
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# train our model
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5')
print("Done")
