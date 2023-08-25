import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import json
import pickle
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)
print(intents)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]

pickle.dump(words, open('words.pkl','wb'))
pickle.dump(classes, open('classes.pkl','wb'))

pattern_strings = [' '.join(pattern) for pattern, _ in documents]

# Creating the bag-of-words representation using CountVectorizer
vectorizer = CountVectorizer(lowercase=True, analyzer='word', tokenizer=nltk.word_tokenize)
X = vectorizer.fit_transform(pattern_strings).toarray()

max_words = X.shape[1]

# Preparing training data
max_words = len(vectorizer.vocabulary_)
training_data = []
for i, (_, intent) in enumerate(documents):
    output_row = [0] * len(classes)
    output_row[classes.index(intent)] = 1
    training_data.append((X[i], output_row))

# Separating training features (X) and labels (Y)
train_x = np.array([x for x, _ in training_data])
train_y = np.array([y for _, y in training_data])

# Creating the model
model = Sequential()
model.add(Dense(128, input_shape=(max_words,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax'))

# Using the legacy version of SGD optimizer
sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Saving the vectorizer's vocabulary
pickle.dump(vectorizer.vocabulary_, open('vectorizer_vocab.pkl', 'wb'))

# Fitting and saving the model
hist = model.fit(train_x, train_y, epochs=300, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("Model created and trained.")