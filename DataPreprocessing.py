# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 01:03:28 2018

@author: Shashank
"""
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import re


def cleanData(data):
    data = data[data['class'] != "Neutral"]
    data['text'] = data['text'].apply(lambda x: x.lower())
    data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
    return data

def tokenPadData(data):
    max_fatures = 2000
    tokenizer = Tokenizer(num_words=max_fatures, split=' ')
    tokenizer.fit_on_texts(data['text'].values)
    X1 = tokenizer.texts_to_sequences(data['text'].values)
    X2 = pad_sequences(X1)
    return X2
