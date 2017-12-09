'''
Get stock close data.
'''
import pandas_datareader as pdr
from datetime import datetime

import numpy as np

def get_data(ticker, start, stop):
    '''
    Gathering time series from yahoo finance.
    
    Parameters
    ----------
    ticker : string
        Stock symbol.
    start : datetime
        Start date.
    stop : datetime
        Stop date.
        
    Returns
    ----------
    array
        Time series of stock value at close.
    '''
    res = pdr.get_data_yahoo(symbols = ticker, start = start, end = stop)
    return np.array(res['Adj Close'])

