from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Normalizer
import numpy as np
from matplotlib import pyplot
from tensorflow.keras.models import Model, load_model
from sklearn.metrics import accuracy_score
from DataProcessing import test_Y, train_Y, test_normX, train_normX, train_X, test_X
from FeatureSelection import train_normX_ETC_FS, test_normX_ETC_FS
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

# Run baseline log reg model - normalised
model = LogisticRegression(max_iter=10000)
# fit model on training set
model.fit(train_normX, train_Y)
# make prediction on test set
yhat = model.predict(test_normX)
# calculate accuracy
acc = accuracy_score(test_Y, yhat)
print("base model with norm:", acc)

# Run log reg model with feature extraction
# load the model from file
encoder = load_model('encoder.h5')
# encode the train data
X_train_encode = encoder.predict(train_normX)
# encode the test data
X_test_encode = encoder.predict(test_normX)
# define the model
model = LogisticRegression(max_iter=10000)
# fit the model on the training set
model.fit(X_train_encode, train_Y)
# make predictions on the test set
yhat = model.predict(X_test_encode)
# calculate classification accuracy
acc = accuracy_score(test_Y, yhat)
print('encoded model:',acc)


# # Run log reg model with feature extraction merged with feature selection
# # load the model from file
# encoder = load_model('encoder.h5')
# # encode the train data, and merge with feature selected data
# X_train_encode = encoder.predict(train_normX)
# X_train_merge = np.column_stack([X_train_encode, train_normX_ETC_FS])
# # encode the test data, and merge with feature selected data
# X_test_encode = encoder.predict(test_normX)
# X_test_merge = np.column_stack([X_test_encode, test_normX_ETC_FS])
# print(X_train_merge.shape)
# print(X_test_merge.shape)
# # define the model
# model = LogisticRegression(max_iter=10000)
# # fit the model on the training set
# model.fit(X_train_merge, train_Y)
# # make predictions on the test set
# yhat = model.predict(X_test_merge)
# # calculate classification accuracy
# acc = accuracy_score(test_Y, yhat)
# print('merged model:',acc)
#
#
# # create Ml model
# model = Sequential()
# model.add(Dense(12, input_dim=152, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# # compile model
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# # fit the model
# hist = model.fit(train_normX, train_Y, epochs=25, batch_size=10, validation_data=(test_normX, test_Y))
# # evaluate the model
# scores = model.evaluate(test_normX, test_Y)
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#
# # plot loss
# pyplot.plot(hist.history['loss'], label='train')
# pyplot.plot(hist.history['val_loss'], label='test')
# # limits = [ 0, 50, 0, .025]
# # pyplot.axis(limits)
# pyplot.legend()
# pyplot.show()
