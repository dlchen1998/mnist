from sklearn import metrics

#AMI

def judger(labels_true,labels_pred):
    score = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
    print(score)