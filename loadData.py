import numpy as np
import index
from sklearn.decomposition import PCA
from PIL import Image
import scipy.io as sio


# 参数

mat = sio.loadmat('.\Data\mnist.mat')

data = mat['data']
label = mat['label']


def getData(threshold,n_components,Current_index):


# 阈值法
   for i in range(10000):
        for j in range(748):
            if data[i][j] > threshold:
                data[i][j] = 1
            else:
                data[i][j] = 0

   trainingset = data[index.train_index[Current_index]]

   pca = PCA(n_components, whiten=True)

   trainingset = pca.fit_transform(trainingset)

   traininglabel = label[0][index.train_index[Current_index]]

   testset = data[index.test_index[Current_index]]

   testset = pca.transform(testset)

   testlabel = label[0][index.test_index[Current_index]]

   return trainingset,traininglabel,testset,testlabel



