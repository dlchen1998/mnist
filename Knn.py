#!/usr/bin/python
# coding=utf-8
#########################################
# kNN: k Nearest Neighbors

#  输入:      newInput:  (1xN)的待分类向量
#             dataSet:   (NxM)的训练数据集
#             labels:     训练数据集的类别标签向量
#             k:         近邻数

# 输出:     可能性最大的分类标签
#########################################

from numpy import *
import operator
import loadData

def gaussian(dist, a=1, b=0, c=0.3):
    return a * math.e ** (-(dist - b) ** 2 / (2 * c ** 2))


# KNN分类算法函数定义
def kNNClassify(newInput, dataSet, labels, k):
    numSamples = dataSet.shape[0]   # shape[0]表示行数

    # # step 1: 计算距离
    # 1、求差
    # 2、对差值平方
    # 3、将平方后的差值累加
    # 4、将上一步骤的值求开方，即得距离
    # tile(A, reps): 构造一个矩阵，通过A重复reps次得到

    diff = tile(newInput, (numSamples, 1)) - dataSet  # 按元素求差值
    #print(dataSet.shape)

    squaredDiff = diff ** 2  # 将差值平方
    squaredDist = sum(squaredDiff, axis = 1)   # 按行累加
    distance = squaredDist ** 0.5  # 将差值平方和求开方，即得距离

    # # step 2: 对距离排序
    # argsort() 返回排序后的索引值
    sortedDist = sort(distance)
    sortedDistIndices = argsort(distance)
    classCount = {} # define a dictionary (can be append element)
    for i in range(k):
        # # step 3: 选择k个最近邻
        voteLabel = labels[sortedDistIndices[i]]

        # # step 4: 计算k个最近邻中各类别出现的次数
        classCount[voteLabel] = classCount.get(voteLabel, 0) + gaussian(sortedDist[i])

    # # step 5: 返回出现次数最多的类别标签
    maxCount = 0
    maxIndex=0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key

    return maxIndex