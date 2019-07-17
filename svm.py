"""
mnist_svm
~~~~~~~~~
使用SVM分类器，从MNIST数据集中进行手写数字识别的分类程序
"""

import loadData
# Third-party libraries
from sklearn import svm
import time
import Judge


clf = svm.SVC()
clf.fit(loadData.trainingset, loadData.traininglabel)
predictions = [int(a) for a in clf.predict(loadData.testset)]

Judge.judger(predictions,loadData.testlabel)
