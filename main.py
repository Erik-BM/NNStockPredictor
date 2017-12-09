'''
Simple stock predictor.
Using previous N days to predict day N+1.
'''

import numpy as np
from sklearn.neural_network import MLPRegressor

from pandas_datareader import data
import pandas as pd

from get_data import get_data

from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt

import sys

#Using NUMBER_OF_DAYS-1 to predict the final day.
NUMBER_OF_DAYS = 5

'''
Number of days back in time used. Weekends and such will be removed. 
This means that length of training set < TRAINING_DAYS.
'''
TRAINING_DAYS = 200

def prepare_training_set(ticker, rand = True):
    '''
    Preparing training set. 
    
    Sliding window chuncks of length NUMBER_OF_DAYS.
    Last element used as label.
    
    Parameters
    ----------
    ticker : string
        Stock symbol. eg. AAPL
    rand : bool
        Randomize output.
        
    Returns
    ----------
    X : array
        Training data. 
    y : array
        Labels.
    '''
    train_start = date.today()- timedelta(days=TRAINING_DAYS)
    train_stop = date.today()- timedelta(days=1)
    
    X = np.array(get_data(ticker, train_start, train_stop))
    
    X = X[len(X) % NUMBER_OF_DAYS:]
   
    #Normalizing data
    X = X - min(X)
    X = X / max(X)
    
    y = []
    X_final = []
    for i in range(len(X) - NUMBER_OF_DAYS):
        x = X[i:i+NUMBER_OF_DAYS]
        y.append(x[-1])
        X_final.append(x[:-1])
    
    y = np.array(y)
    X = np.array(X_final)
    
    if rand == True:
        seq = np.random.choice(len(X), len(X))
    
        y = y[seq]
        X = X[seq]
    
    return X, np.ravel(y)

def train_network(X, y):
    '''
    Training network.
    
    Training neural network using regression.
    
    Parameters
    ----------
    X : array
        Training data. (#samples, #features)
    y : array
        Labels. (#samples,)
        
    Returns
    ----------
    clf : MLPRegressor object
        Trained model.
    '''
    
    N = len(X)
    m = 1
    num_neurons_1 = int(round(np.sqrt((m + 2) * N) + 2 * np.sqrt(N/(m + 2))))
    num_neurons_2 = int(round(m * np.sqrt(N/(m + 2))))

    clf = MLPRegressor(learning_rate = 'adaptive', alpha = 1e-5, hidden_layer_sizes=(num_neurons_1,num_neurons_2), verbose = True, max_iter = 1000, tol = 1e-6)

    clf.fit(X, y)
    
    return clf


def main(tickers):
    
    for ticker in tickers:
        
        X, y = prepare_training_set(ticker, rand = False)
        
        net = train_network(X, y)
        
        predicted_y = []
        for x in X:
            out = net.predict(x.reshape(1,-1))
            predicted_y.append(out)
            
        y = np.ravel(y)
        predicted_y = np.ravel(predicted_y)
        
        print(len(y),len(predicted_y))
        
        predplot, = plt.plot(predicted_y, color='red')
        trueplot, = plt.plot(y, color = 'green')
        plt.title(ticker)
        plt.legend([predplot, trueplot], ['Predicted', 'True'])
        plt.savefig(ticker + '.png')
        plt.close()
    
if __name__ == "__main__":
    args = sys.argv
    if len(args) < 2:
        print('Usage: Python3 main.py STOCK1 STOCK2 ...')
        exit()
    main(args[1:])
