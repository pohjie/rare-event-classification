import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
from pylab import rcParams

import tensorflow as tf
from keras import optimizers, Sequential
from keras.models import Model
from keras.utils import plot_model
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed
from keras.callbacks import ModelCheckpoint, TensorBoard

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(123)

from sklearn.model_selection import train_test_split

SEED = 123 #used to help randomly select the data points
DATA_SPLIT_PCT = 0.2

rcParams['figure.figsize'] = 8, 6
LABELS = ["Normal","Break"]

df = pd.read_csv("processminer-rare-event-mts - data.csv") 
print(df.head(n=5))  # visualize the data.

sign = lambda x: (1, -1)[x < 0]

def curve_shift(df, shift_by):
    '''
    This function will shift the binary labels in a dataframe.
    The curve shift will be with respect to the 1s. 
    For example, if shift is -2, the following process
    will happen: if row n is labeled as 1, then
    - Make row (n+shift_by):(n+shift_by-1) = 1.
    - Remove row n.
    i.e. the labels will be shifted up to 2 rows up.
    
    Inputs:
    df       A pandas dataframe with a binary labeled column. 
             This labeled column should be named as 'y'.
    shift_by An integer denoting the number of rows to shift.
    
    Output
    df       A dataframe with the binary labels shifted by shift.
    '''

    vector = df['y'].copy()
    for s in range(abs(shift_by)):
        tmp = vector.shift(sign(shift_by))
        tmp = tmp.fillna(0)
        vector += tmp
    labelcol = 'y'
    # Add vector to the df
    df.insert(loc=0, column=labelcol+'tmp', value=vector)
    # Remove the rows with labelcol == 1.
    df = df.drop(df[df[labelcol] == 1].index)
    # Drop labelcol and rename the tmp col as labelcol
    df = df.drop(labelcol, axis=1)
    df = df.rename(columns={labelcol+'tmp': labelcol})
    # Make the labelcol binary
    df.loc[df[labelcol] > 0, labelcol] = 1

    return df

input_y = df['y']

# Check if the shifting of data is done correctly
print('Before shifting') # focus on positive labelled rows
one_indexes = df.index[df['y'] == 1]
print(df.iloc[(np.where(np.array(input_y) == 1)[0][0]-5):(np.where(np.array(input_y) == 1)[0][0]+1), ])

# Shift the response column y by 2 rows to do a 4-min ahead prediction
df = curve_shift(df, shift_by=-2)

print('After shifting') # Validating if the shift is right
print(df.iloc[(one_indexes[0]-4):(one_indexes[0]+1), 0:5].head(n=5))

# Extract features and responses
df = df.drop(columns=['time'])
input_X = df.loc[:, df.columns != 'y'].values # converts df to a numpy array
input_y = df['y'].values

n_features = input_X.shape[1]

def temporalize(input_X, input_y, lookback):
    X = []
    y = []

    for i in range(len(input_X)-lookback-1):
        t = []
        for j in range(1, lookback+1):
            # Gather past records up to the lookback period
            t.append(input_X[[(i+j+1)], :])

        X.append(t)
        y.append(input_y[i+lookback+1])

    return X, y

print('First instance of y=1 in the original data')
print(df.iloc[(np.where(np.array(input_y) == 1)[0][0] - 5):(np.where(np.array(input_y) == 1)[0][0]+1), ])

lookback = 5 # equivalent to 10 min of past data
# Temporalize the data
X, y = temporalize(input_X, input_y, lookback)

print('For the same instance of y = 1, we are keeping past 5 samples in the 3D predictor array, X.')
print(pd.DataFrame(np.concatenate(X[np.where(np.array(y) == 1) [0][0]], axis=0)))

# Spilt into train, test valid set
X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=DATA_SPLIT_PCT, random_state=SEED)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=DATA_SPLIT_PCT, random_state=SEED)

X_train_y0 = X_train[y_train==0]
X_train_y1 = X_train[y_train==1]

X_valid_y0 = X_valid[y_valid==0]
X_valid_y1 = X_valid[y_valid==1]

# Reshape X into required 3D dimensions
X_train = X_train.reshape(X_train.shape[0], lookback, n_features)
X_train_y0 = X_train_y0.reshape(X_train_y0.shape[0], lookback, n_features)
X_train_y1 = X_train_y1.reshape(X_train_y1.shape[0], lookback, n_features)

X_valid = X_valid.reshape(X_valid.shape[0], lookback, n_features)
X_valid_y0 = X_valid_y0.reshape(X_valid_y0.shape[0], lookback, n_features)
X_valid_y1 = X_valid_y1.reshape(X_valid_y1.shape[0], lookback, n_features)

X_test = X_test.reshape(X_test.shape[0], lookback, n_features)

# Flatten function- inverse of temporalise
def flatten(X):
    '''
    Flatten a 3D array
    Input
    X: A 3D array for LSTM, where the dimensions are num_samples * timesteps * n_features
    Output
    flattened_X: A 2D array, where the dimensions are num_samples * n_features
    '''
    flattened_X = np.empty((X.shape[0], X.shape[2])) # n_samples * n_features array
    for i in range(X.shape[0]):
        flattened_X[i] = X[i, (X.shape[1] - 1), :]
    return flattened_X

def scale(X, scaler): 
    '''
    Scale 3D array.

    Inputs
    X: A 3D array for LSTM, where the array is num_samples, * timesteps * features
    scaler: A scaler object
    Output
    scaled_X: Scaled 3D array
    '''
    for i in range(X.shape[0]):
        X[i, :, :] = scaler.transform(X[i, :, :])

    return X

# Initialize a scaler using the training data.
scaler = StandardScaler().fit(flatten(X_train_y0))

X_train_y0_scaled = scale(X_train_y0, scaler)

# scale validation and test set
X_valid_scaled = scale(X_valid, scaler)
X_valid_y0_scaled = scale(X_valid_y0, scaler)

X_test_scaled = scale(X_test, scaler)

# LSTM Autoencoder Training
timesteps = X_train_y0_scaled.shape[1]
n_features = X_train_y0_scaled.shape[2]

num_epochs = 500
batch_size = 64
lr = 0.0001