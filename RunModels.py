from sklearn.linear_model import LogisticRegression
import numpy as np
from tensorflow.keras.models import Model, load_model
from sklearn.metrics import accuracy_score
from DataProcessing import test_Y, train_Y, test_normX, train_normX
from FeatureSelection import train_normX_ETC_FS

# Run baseline log reg model
model = LogisticRegression(max_iter=10000)
# fit model on training set
model.fit(train_normX, train_Y)
# make prediction on test set
yhat = model.predict(test_normX)
# calculate accuracy
acc = accuracy_score(test_Y, yhat)
print("base model:", acc)

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


# Run log reg model with feature extraction merged with feature selection
# load the model from file
encoder = load_model('encoder.h5')
# encode the train data, and merge with feature selected data
X_train_encode = encoder.predict(train_normX)
X_train_merge = np.column_stack([X_train_encode, train_normX_ETC_FS])
print(X_train_merge.shape)
# encode the test data, and merge with feature selected data
X_test_encode = encoder.predict(test_normX)
print(X_test_encode.shape)
# define the model
model = LogisticRegression(max_iter=10000)
# fit the model on the training set
model.fit(X_train_encode, train_Y)
# make predictions on the test set
yhat = model.predict(X_test_encode)
# calculate classification accuracy
acc = accuracy_score(test_Y, yhat)
print('encoded model:',acc)
