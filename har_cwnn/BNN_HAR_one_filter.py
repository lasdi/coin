from __future__ import print_function
import pickle
import numpy as np
from numpy import mean, std, dstack, stack, hstack, vstack
from pandas import read_csv
from sklearn.preprocessing import StandardScaler

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

# load HAR dataset
trainX, trainy, testX, testy = load_dataset()
# scale data
trainX, testX = scale_data(trainX, testX)

# transform trainX into how input data looks to the filters
# stride = 1, padding = 1, filter_size = k = 3
k = 5 
padding = np.zeros(9)
trainX_new = list()

for i in range(0,len(trainX)):
    sample = trainX[i,:,:]
    # pad the first and last row of the sample
    row_first = 0
    sample = np.insert(sample, row_first, [padding], axis=0)
    sample = np.insert(sample, row_first, [padding], axis=0)
    row_last = sample.shape[0]
    sample = np.insert(sample, row_last, [padding], axis=0)
    sample = np.insert(sample, row_last, [padding], axis=0)
    # run the sliding filter
    for j in range(len(sample)-k+1):
        mat = sample[j:j+k, :]
        trainX_new.append(mat)
trainX_new = stack(trainX_new)
print("New dataset input shape = ", trainX_new.shape)

# save the new trainX dataset
with open('./WNN_dataset/'+'trainX.pkl', 'wb') as outp:
    pickle.dump(trainX_new, outp, pickle.HIGHEST_PROTOCOL)

# get activations for labels y
with open('_act1_out_train.pkl', 'rb') as inp:
    act1_out_train = pickle.load(inp)

trainy_new = []

for i in range(len(act1_out_train)):
    label = act1_out_train[i,:,:]
    # extract first filter activation outputs
    label = label[:,0]
    trainy_new.append(label)
trainy_new = hstack(trainy_new)
trainy_new = np.reshape(trainy_new, (941056,-1))
print("New dataset label shape = ", trainy_new.shape) 

# save the new labels for the dataset
with open('./WNN_dataset/'+'trainy.pkl', 'wb') as outp:
    pickle.dump(trainy_new, outp, pickle.HIGHEST_PROTOCOL)
