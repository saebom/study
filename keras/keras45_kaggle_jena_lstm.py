import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import LSTM, Dense, Input
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import tensorflow as tf


#1. 데이터
path = './_data/kaggle_jena/'
dataset = pd.read_csv(path + 'jena_climate_2009_2016.csv', 
                        index_col=0)

print(dataset.shape)    # (420551, 14)
print(dataset.describe)



