
import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout,SpatialDropout1D,GlobalMaxPooling1D

from keras.models import Sequential
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import string

#Text Preprocessing function

tokenizer1 = RegexpTokenizer('[a-zA-Z]\w+\'?\w*')
from nltk.corpus import stopwords
nltk.download('stopwords') 
def clean_str(doc):
    # split into tokens by white space
    tokens = tokenizer1.tokenize(doc)
    # remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out short tokens
    tokens = [word.lower() for word in tokens if len(word) > 1]

    return " ".join(tokens)

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2



# reading data


df = pd.read_csv('ex3_train_data.csv')
df = df.dropna()
df = df.reset_index(drop=True)

sentences=[]
for i in df['sentence'].values:
    sentences.append(clean_str(i))

y=df['label'].values
sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.2, random_state=1000)

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences_train)
X_train = tokenizer.texts_to_sequences(sentences_train)
X_test = tokenizer.texts_to_sequences(sentences_test)

vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index


from keras.preprocessing.sequence import pad_sequences

maxlen = 100

X_train = pad_sequences(X_train, padding='post', maxlen=100)
X_test = pad_sequences(X_test, padding='post', maxlen=100)


embedding_dim = 100

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=100))
model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train,
                    epochs=10,
                    validation_data=(X_test, y_test),
                    batch_size=10)

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))


test_df = pd.read_csv('ex3_test_data.csv')
test_df.head()

sentences_p=[]
for i in test_df['sentence'].values:
    sentences_p.append(clean_str(i))

X_predict = tokenizer.texts_to_sequences(sentences_p)

X_predict = pad_sequences(X_predict, padding='post', maxlen=100)
preds=model.predict(X_predict)

ans=[]
for i in preds:
  if i>0.5:
    ans.append(1)
  else:
    ans.append(0)

Test_pred = pd.DataFrame({"id":test_df['id'],"label":ans})
Test_pred.to_csv('Results.csv') 

