# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 01:11:32 2018

@author: Shashank
"""
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
import numpy as np


class Model:
    def __init__(self,w):
        self.model=self.getModel(w)

    def trainModel(self,X_train, Y_train, batch_size):
        self.model.fit(X_train, Y_train, epochs = 7, batch_size=batch_size, verbose = 2, validation_split=0.10)

    def testModel(self, X_test, Y_test, batch_size):
        score,acc = self.model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
        print("score: %.2f" % (score))
        print("acc: %.2f" % (acc))
    
    def getModel(self,w):
        embed_dim = 12
        lstm_out = 196
        
        model = Sequential()
        model.add(Embedding(2000, embed_dim,input_length = w))
        model.add(SpatialDropout1D(0.4))
        model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(2,activation='softmax'))
        model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
        print(model.summary())
        return model
    
    def saveModel(self,name):
        self.model.save('SavedModels/'+name+'.h5')