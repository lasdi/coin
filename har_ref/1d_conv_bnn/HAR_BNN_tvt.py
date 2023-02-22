# v2 - Cross validation + different accuracy evaluations + export layer values; kept train, validation and test data seperate

# Trains a simple binarize CNN on the HAR dataset

from __future__ import print_function
import numpy as np
import keras.backend as K
import pickle
from numpy import mean, std, dstack
from pandas import read_csv
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, Activation
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.callbacks import LearningRateScheduler
from keras.utils import to_categorical, np_utils
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from binary_ops import binary_tanh as binary_tanh_op
from binary_layers import BinaryDense, BinaryConv1D

np.random.seed(1337)  # for reproducibility

def binary_tanh(x):
    return binary_tanh_op(x)

# load a single file as a numpy array
def load_file(filepath):
    dataframe = read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values
 
# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
    # stack group so that features are the 3rd dimension
    loaded = dstack(loaded)
    return loaded
 
# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
    filepath = prefix + group + '/Inertial Signals/'
    # load all 9 files as a single array
    filenames = list()
    # total acceleration
    filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
    # body acceleration
    filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
    # body gyroscope
    filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
    # load input data
    X = load_group(filenames, filepath)
    # load class output
    y = load_file(prefix + group + '/y_'+group+'.txt')
    return X, y
 
# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
    # load all train
    trainX, trainy = load_dataset_group('train', prefix + 'HARDataset/')
    # load all test
    testX, testy = load_dataset_group('test', prefix + 'HARDataset/')
    print(trainX.shape, trainy.shape, testX.shape, testy.shape)
    return trainX, trainy, testX, testy
 
# standardize data
def scale_data(trainX, testX):
    # remove overlap
    cut = int(trainX.shape[1] / 2)
    longX = trainX[:, -cut:, :]
    # flatten windows
    longX = longX.reshape((longX.shape[0] * longX.shape[1], longX.shape[2]))
    # flatten train and test
    flatTrainX = trainX.reshape((trainX.shape[0] * trainX.shape[1], trainX.shape[2]))
    flatTestX = testX.reshape((testX.shape[0] * testX.shape[1], testX.shape[2]))
    # standardize
    s = StandardScaler()
    # fit on training data
    s.fit(longX)
    # apply to training and test data
    longX = s.transform(longX)
    flatTrainX = s.transform(flatTrainX)
    flatTestX = s.transform(flatTestX)
    # reshape
    flatTrainX = flatTrainX.reshape((trainX.shape))
    flatTestX = flatTestX.reshape((testX.shape))
    return flatTrainX, flatTestX

H = 1.
kernel_lr_multiplier = 'Glorot'

# BNN
batch_size = 50
epochs = 1
kernel_size = 3
pool_size = 2
classes = 6
use_bias = False

# learning rate schedule
lr_start = 1e-3
lr_end = 1e-6
lr_decay = (lr_end / lr_start)**(1. / epochs)

# BN
epsilon = 1e-6
momentum = 0.9

# dropout
p1 = 0.25
p2 = 0.5

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# fit and evaluate a model
def evaluate_model():
    # load data
    trainX, trainy, testX, testy = load_dataset()
    # scale data
    trainX, testX = scale_data(trainX, testX)

    # zero-offset class values
    testy_int = testy - 1
    # one hot encode y
    testy_int = to_categorical(testy_int)

    # define 5-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cvscores = []
    
    for train, val in kfold.split(trainX, trainy):

        # zero-offset class values
        trainy_int = trainy - 1
        # one hot encode y
        trainy_int = to_categorical(trainy_int)

        n_timesteps, n_features, n_outputs = trainX[train].shape[1], trainX[train].shape[2], trainy_int[train].shape[1]

        model = Sequential()
    
        # conv1
        model.add(BinaryConv1D(filters=1, kernel_size=kernel_size, input_shape=(n_timesteps, n_features),
                       H=H, kernel_lr_multiplier=kernel_lr_multiplier, 
                       padding='same', use_bias=use_bias, name='conv1'))
        model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn1'))
        model.add(Activation(binary_tanh, name='act1'))
    
        # conv2
        model.add(BinaryConv1D(filters=128, kernel_size=kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                       padding='same', use_bias=use_bias, name='conv2'))
        model.add(MaxPooling1D(pool_size=pool_size, name='pool2'))
        model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn2'))
        model.add(Activation(binary_tanh, name='act2'))

        model.add(Flatten())
        model.add(BinaryDense(classes, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias, name='dense6'))
        model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn6'))

        opt = Adam(lr=lr_start) 
        model.compile(loss='squared_hinge', optimizer=opt, metrics=['acc'])
        model.summary()

        lr_scheduler = LearningRateScheduler(lambda e: lr_start * lr_decay ** e)
        history = model.fit(trainX[train], trainy_int[train], batch_size=batch_size, epochs=epochs,
                    verbose=1, validation_data=(trainX[val], trainy_int[val]), callbacks=[lr_scheduler])
        score = model.evaluate(trainX[val], trainy_int[val], verbose=0)
        print('Accuracy:', score[1])
        cvscores.append(score[1])
        y_pred = model.predict(testX)
        y_pred = np.argmax(y_pred, axis=1) + 1
 
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores)*100, np.std(cvscores)*100))

    # keras
    score = model.evaluate(testX, testy_int, verbose=0)
    print('Keras accuracy on test data:', score[1])

    # scikit learn
    scikit_learn_score = accuracy_score(testy, y_pred)
    print('Scikit learn accuracy on test data:', scikit_learn_score)
    cm = confusion_matrix(testy, y_pred)  
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()  

    with open('./bnn_har.pkl', 'wb') as outp:
        pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)

evaluate_model()

with open('./bnn_har.pkl', 'rb') as inp:
    model = pickle.load(inp)

def get_layer_data (model, X, layer_name):    
    for i in range (len(model.layers)):
        if layer_name == model.layers[i].name:
            print('Layer: ' + model.layers[i].name, end=' ')             
            layer_func = K.function([model.layers[0].input], [model.layers[i].output])
            layer_data = layer_func([X])[0]
            print(layer_data.shape)

    return layer_data

#load data
trainX, trainy, testX, testy = load_dataset()
# scale data
trainX, testX = scale_data(trainX, testX)

act1_out_train = get_layer_data (model, trainX, 'act1')
print(act1_out_train[0])
print(act1_out_train[0].shape)
act2_out_train = get_layer_data (model, trainX, 'act2')
bn6_out_train = get_layer_data (model, trainX, 'bn6')

act1_out_test = get_layer_data (model, testX, 'act1')
act2_out_test = get_layer_data (model, testX, 'act2')
bn6_out_test = get_layer_data (model, testX, 'bn6')

#print('activation 1 outputs:', act1_out_train)
#print('###############################')
#print('activation 1 outputs:', act1_out_test)
#print('###############################')
#print('activation 2 outputs:', act2_out)
#print('###############################')
#print('batch normalization 6 outputs:', bn6_out)
#print('###############################')
        
with open('./layer_data/'+'_act1_out_train.pkl', 'wb') as outp:
    pickle.dump(act1_out_train, outp, pickle.HIGHEST_PROTOCOL)        
with open('./layer_data/'+'_act2_out_train.pkl', 'wb') as outp:
    pickle.dump(act2_out_train, outp, pickle.HIGHEST_PROTOCOL)        
with open('./layer_data/'+'_bn6_out_train.pkl', 'wb') as outp:
    pickle.dump(bn6_out_train, outp, pickle.HIGHEST_PROTOCOL)
with open('./layer_data/'+'_act1_out_test.pkl', 'wb') as outp:
    pickle.dump(act1_out_test, outp, pickle.HIGHEST_PROTOCOL)        
with open('./layer_data/'+'_act2_out_test.pkl', 'wb') as outp:
    pickle.dump(act2_out_test, outp, pickle.HIGHEST_PROTOCOL)        
with open('./layer_data/'+'_bn6_out_test.pkl', 'wb') as outp:
    pickle.dump(bn6_out_test, outp, pickle.HIGHEST_PROTOCOL)  