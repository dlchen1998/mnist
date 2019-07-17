import numpy as np
from sklearn.decomposition import PCA
from PIL import Image
import scipy.io as sio


def read_data(path):
    mat = sio.loadmat(path)
    return mat['data'], mat['label']


def read_index(path):
    mat = sio.loadmat(path)
    return mat['train_index'], mat['test_index']


def pre_processing(data, threshold=0):
    # 阈值法
    for picture in data:
        for i in range(picture.size):
            if picture[i] > threshold:
                picture[i] = 1
            else:
                picture[i] = 0

    # pca
    pca = PCA(n_components=30, whiten=True)
    data = pca.fit_transform(data)

    return data


def get_test_data(data, label, current_train_index, current_test_index):
    train_label = label[0][current_train_index]
    test_label = label[0][current_test_index]

    train_set = data[current_train_index]
    test_set = data[current_test_index]

    return train_set, test_set, train_label, test_label


if __name__ == '__main__':
    print("test of loadData.py".center(80, '-'))
    current_index = 1
    print(current_index)
    data_path = 'data\\mnist.mat'
    index_path = 'data\\index.mat'

    data, label = read_data(data_path)
    train_index, test_index = read_index(index_path)
    print(data.shape)
    print(label.shape)
    print(train_index.shape)
    print(test_index.shape)

    data = pre_processing(data)
    print(data.shape)

    train_set, test_set, train_label, test_label = get_test_data(data, label, train_index[current_index], test_index[current_index])
    print(train_set.shape)
    print(test_set.shape)
    print(train_label.shape)
    print(test_label.shape)
