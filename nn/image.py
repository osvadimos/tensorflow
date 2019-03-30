#from scipy.misc import imread, imresize
import pickle
import numpy as np
from nn.autoencoder import Autoencoder


def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict

path = '/home/vadim/tmp/python/cifar-10-batches-py/'

name = unpickle(path + 'batches.meta')['label_names']
data, labels = [], []
for i in range(1, 6):
    filename = path + 'data_batch_' + str(i)
    batch_data = unpickle(filename)
    if len(data) > 0:
        data = np.vstack((data, batch_data['data']))
        labels = np.hstack((labels, batch_data['labels']))
    else:
        data = batch_data['data']
        labels = batch_data['labels']


def grayscale(a):
    return a.reshape(a.shape[0], 3, 32, 32).mean(1).reshape(a.shape[0], -1)

data = grayscale(data)

x = np.matrix(data)
y = np.array(labels)

horse_indices = np.where(y == 7)[0]
horse_x = x[horse_indices]

print(np.shape(horse_x))

input_dim = np.shape(horse_x)[1]
hidden_dim = 100
ae = Autoencoder(input_dim, hidden_dim)
ae.train(horse_x)


exit()
