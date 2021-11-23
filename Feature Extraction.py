from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
from pandas import read_csv
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot
import numpy as np

#Load Files
filename = 'train_imperson_without4n7_balanced_data.csv'
names = ['1',	'2',	'3',	'5',	'6',	'8',	'9',	'10',	'11',	'12',	'13',	'14',	'15',	'16',	'17',	'18',	'19',	'20',	'21',	'22',	'23',	'24',	'25',	'26',	'27',	'28',	'29',	'30',	'31',	'32',	'33',	'34',	'35',	'36',	'37',	'38',	'39',	'40',	'41',	'42',	'43',	'44',	'45',	'46',	'47',	'48',	'49',	'50',	'51',	'52',	'53',	'54',	'55',	'56',	'57',	'58',	'59',	'60',	'61',	'62',	'63',	'64',	'65',	'66',	'67',	'68',	'69',	'70',	'71',	'72',	'73',	'74',	'75',	'76',	'77',	'78',	'79',	'80',	'81',	'82',	'83',	'84',	'85',	'86',	'87',	'88',	'89',	'90',	'91',	'92',	'93',	'94',	'95',	'96',	'97',	'98',	'99',	'100',	'101',	'102',	'103',	'104',	'105',	'106',	'107',	'108',	'109',	'110',	'111',	'112',	'113',	'114',	'115',	'116',	'117',	'118',	'119',	'120',	'121',	'122',	'123',	'124',	'125',	'126',	'127',	'128',	'129',	'130',	'131',	'132',	'133',	'134',	'135',	'136',	'137',	'138',	'139',	'140',	'141',	'142',	'143',	'144',	'145',	'146',	'147',	'148',	'149',	'150',	'151',	'152',	'153',	'154',	'155',]
dataframe = read_csv(filename, names=names, skiprows=1)
array = dataframe.values

filename2 = 'test_imperson_without4n7_balanced_data.csv'
dataframe2 = read_csv(filename2, names=names, skiprows=1)
array2 = dataframe2.values

# Separate array into input and output components
train_X = array[:,0:152]
train_Y = array[:,152]
test_X = array2[:,0:152]
test_Y = array2[:,152]

# Normalise X Variables
scaler = Normalizer().fit(train_X)
train_normX = scaler.transform(train_X)

scaler2 = Normalizer().fit(test_X)
test_normX = scaler2.transform(test_X)

# define encoder
n_inputs = train_normX.shape[1]
visible = Input(shape=(n_inputs,))
# encoder level 1
e = Dense(n_inputs*2)(visible)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# encoder level 2
e = Dense(n_inputs)(e)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# bottleneck
n_bottleneck = round(float(n_inputs) / 10.0)
bottleneck = Dense(n_bottleneck)(e)

# define decoder, level 1
d = Dense(n_inputs)(bottleneck)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# decoder level 2
d = Dense(n_inputs*2)(d)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# output layer
output = Dense(n_inputs, activation='linear')(d)

# define autoencoder model
model = Model(inputs=visible, outputs=output)

# compile autoencoder model
model.compile(optimizer='adam', loss='mse')

# #plot autoencoder
# plot_model(model, 'autoencoder_no_compress.png', show_shapes=True)

# fit the autoencoder model to reconstruct input
hist = model.fit(train_normX, train_normX, epochs=50, batch_size=16, verbose=2,
                    validation_data=(test_normX, test_normX))

# plot loss
pyplot.plot(hist.history['loss'], label='train')
pyplot.plot(hist.history['val_loss'], label='test')
limits = [ 0, 50, 0, .025]
pyplot.axis(limits)
pyplot.legend()
pyplot.show()

# save the encoder to file
encoder = Model(inputs=visible, outputs=bottleneck)
encoder.save('encoder.h5')


# Run baseline log reg model
model = LogisticRegression(max_iter=10000)
# fit model on training set
model.fit(train_normX, train_Y)
# make prediction on test set
yhat = model.predict(test_normX)
# calculate accuracy
acc = accuracy_score(test_Y, yhat)
print(acc)


# load the model from file
encoder = load_model('encoder.h5')
# encode the train data
X_train_encode = encoder.predict(train_normX)
# encode the test data
X_test_encode = encoder.predict(X_test)
# define the model
model = LogisticRegression(max_iter=10000)
# fit the model on the training set
model.fit(X_train_encode, train_Y)
# make predictions on the test set
yhat = model.predict(X_test_encode)
# calculate classification accuracy
acc = accuracy_score(test_Y, yhat)
print(acc)

