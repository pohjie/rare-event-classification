# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
from pylab import rcParams

import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

SEED = 123
DATA_SPLIT_PCT = 0.2

rcParams['figure.figsize'] = 8, 6
LABELS = ['Normal', 'Break']

# Data preprocessing
df = pd.read_csv('processminer-rare-event-mts - data.csv')

# UDF for shifting the labels up
sign = lambda x: (1, -1) [x < 0]

def curve_shift(df, shift_by):
	'''
	This function will shift the binary labels in a dataframe
	Curve shift will be with respect to the 1s
	For example, if shift is -2, the following process will happen:
	if row n is labelled as 1, then
	- Make row (n + shift_by):(n+shift_by -1) = 1
	- Remove row n.
	i.e. labels will be shifted up to 2 rows up
	'''

	vector = df['y'].copy()
	for s in range(abs(shift_by)):
		tmp = vector.shift(sign(shift_by))
		tmp = tmp.fillna(0)
		vector += tmp
	labelcol = 'y'

	# Add vector to the df
	df.insert(loc=0, column=labelcol+'tmp', value=vector)

	# Remove the rows with labelcol == 1
	df = df.drop(df[df[labelcol] == 1].index)

	# Drop labelcol and rename the tmp col as labelcol
	df = df.drop(labelcol, axis=1)
	df = df.rename(columns={labelcol+'tmp': labelcol})

	# Make labelcol binary
	df.loc[df[labelcol] > 0, labelcol] = 1

	return df

# Remove time column and categorical columns
df = df.drop(['time', 'x28', 'x61'], axis=1)

df_train, df_test = train_test_split(df, test_size=DATA_SPLIT_PCT,
									 random_state=SEED)
df_train, df_valid = train_test_split(df_train, test_size=DATA_SPLIT_PCT,
									  random_state=SEED)

df_train_0 = df_train.loc[df['y'] == 0]
df_train_1 = df_train.loc[df['y'] == 1]

df_train_0_x = df_train_0.drop(['y'], axis=1)
df_train_1_x = df_train_1.drop(['y'], axis=1)

df_valid_0 = df_valid.loc[df['y'] == 0]
df_valid_1 = df_valid.loc[df['y'] == 1]
df_valid_0_x = df_valid_0.drop(['y'], axis=1)
df_valid_1_x = df_valid_1.drop(['y'], axis=1)

df_test_0 = df_test.loc[df['y'] == 0]
df_test_1 = df_test.loc[df['y'] == 1]
df_test_0_x = df_test_0.drop(['y'], axis=1)
df_test_1_x = df_test_0.drop(['y'], axis=1)

# Standardization
scaler = StandardScaler().fit(df_train_0_x)

df_train_0_x_rescaled = scaler.transform(df_train_0_x)
df_valid_0_x_rescaled = scaler.transform(df_valid_0_x)
df_valid_x_rescaled = scaler.transform(df_valid.drop(['y'], axis=1))

df_test_0_x_rescaled = scaler.transform(df_test_0_x)
df_test_x_rescaled = scaler.transform(df_test.drop(['y'], axis=1))

# Initialise Autoencoder architecture
num_epoch = 1000
batch_size = 128
input_dim = df_train_0_x_rescaled.shape[1] # num of predictor variables
encoding_dim = 32
hidden_dim = int(encoding_dim / 2)
learning_rate = 1e-3

input_layer = Input(shape=(input_dim, ))
encoder = Dense(encoding_dim, activation='tanh',
				activity_regularizer=regularizers.l1(learning_rate))(input_layer)
encoder = Dense(hidden_dim, activation='relu')(encoder)
decoder = Dense(hidden_dim, activation='tanh')(encoder)
decoder = Dense(input_dim, activation='relu')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)

# Training
autoencoder.compile(metrics=['accuracy'], loss='mean_squared_error',
					optimizer='adam')
cp = ModelCheckpoint(filepath='autoencoder_classifier.h5',
					 save_best_only=True, verbose=0)
tb = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True,
				 write_images=True)
history = autoencoder.fit(df_train_0_x_rescaled, df_train_0_x_rescaled,
						  epochs=num_epoch, batch_size=batch_size, shuffle=True,
						  validation_data=(df_valid_0_x_rescaled, df_valid_0_x_rescaled),
						  verbose=1, callbacks=[cp, tb]).history

valid_x_pred = autoencoder.predict(df_valid_x_rescaled)
mse = np.mean(np.power(df_valid_x_rescaled - valid_x_pred, 2), axis=1)
error_df = pd.DataFrame({'Reconstruction_error': mse, 'True_class': df_valid['y']})

precision_rt, recall_rt, threshold_rt = precision_recall_curve(error_df.True_class,
															   error_df.Reconstruction_error)
plt.plot(threshold_rt, precision_rt[1:], label='Precision', linewidth=5)
plt.plot(threshold_rt, recall_rt[1:], label='Recall', linewidth=5)
plt.title('Precision and recall for different threshold values')
plt.xlabel('Threshold')
plt.ylabel('Precision/Recall')
plt.legend()
plt.show()

# Perform classification on test data
test_x_pred = autoencoder.predict(df_test_x_rescaled)
mse = np.mean(np.power(df_test_x_rescaled - test_x_pred, 2), axis=1)
error_df_test = pd.DataFrame({'Reconstruction_error': mse, 'True_class': df_test['y']})
error_df_test = error_df_test.reset_index()

threshold_fixed = 0.85
groups = error_df_test.groupby('True_class')

fig, ax = plt.subplots()

for name, group in groups:
	ax.plot(group.index, group.Reconstruction_error, marker='o',
			ms=3.5, linestyle='', label='Break' if name==1 else "Normal")
ax.hlines(threshold_fixed, ax.get_xlim()[0], ax.get_xlim()[1],
		  colors='r', zorder=100, label='Threshold')
ax.legend()

plt.title('Reconstruction error for different classes')
plt.ylabel('Reconstruction error')
plt.xlabel('Data point index')
plt.show();

# Confusion matrix
pred_y = [1 if e > threshold_fixed else 0 for e in error_df.Reconstruction_error.values]

conf_matrix = confusion_matrix(error_df.True_class, pred_y)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt='d');
plt.title('Confusion matrix')
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()