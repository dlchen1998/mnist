import loadData
import Knn
import datetime
import Judge
from loadData import getData
from sklearn import svm
import matplotlib.pyplot as plt

threshold = 0

#components = 20

com_number = list()
thre_number = list()
accuracy = list()

eff_index=0

for components in range(50):

  threshold += 0.025

  com_number.append(components)
  thre_number.append(threshold)

  for i in range(5):

    total_count = 0

    trainingset,traininglabel,testset,testlabel = getData(0,components,i)

    t1 = datetime.datetime.now()

    label_pred = list()

    success = 0
    fail = 0

    for i in range(4000):
      label_pred.append(Knn.kNNClassify(testset[i],trainingset,traininglabel,3))
      if(label_pred[i]==testlabel[i]):
        success+=1

      else:
        fail+=1

    score = success/(success+fail)
    print(score)
    total_count += score

    t2 = datetime.datetime.now()

    print(t2-t1)

  accuracy.append(total_count / 5)

plt.scatter(com_number,accuracy)
plt.plot(com_number,accuracy)

plt.show()

