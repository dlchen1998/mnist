import scipy.io as sio
import numpy as np
mat = sio.loadmat('.\Data\index.mat')
train_index = mat['train_index']
test_index = mat['test_index']



print(train_index.shape)
print(test_index.shape)
