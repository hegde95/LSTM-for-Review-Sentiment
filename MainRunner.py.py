# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 01:06:21 2018

@author: Shashank
"""

import pandas as pd
from sklearn.model_selection import train_test_split

import DataPreprocessing as dp
import model as mod

def load_dataCSV():
    df = pd.read_csv('Data/sentiment_analysis.csv')
    return df


def main():
    data = load_dataCSV()

    data = dp.cleanData(data)
    X = dp.tokenPadData(data)   
    
    model = mod.Model(X.shape[1])
    
    Y = pd.get_dummies(data['class']).values
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)
    
    batch_size = 32
    model.trainModel(X_train,Y_train,batch_size)
    model.testModel(X_test, Y_test, batch_size)

    mod.saveModel(model,'lstm3')
    
if __name__ == '__main__':
    main()