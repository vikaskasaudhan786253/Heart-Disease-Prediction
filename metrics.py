from decisiontree import Decision_Tree
from randomforest import Random_Forest
from logisticregression import Logistic_Regression



class Metrics:
    def __init__(self):
        self.dt = Decision_Tree()
        self.rf = Random_Forest()
        self.lr = Logistic_Regression()
        self.accuracy = []
        self.precision = []
        self.f1_score = []
        self.recall = []
        self.roc_auc = []

    def Accuracy(self):
        self.accuracy.append(self.lr.accuracy())
        self.accuracy.append(self.dt.accuracy())
        self.accuracy.append(self.rf.accuracy())
        return self.accuracy

    def Precision(self):
        self.precision.append(self.lr.precision())
        self.precision.append(self.dt.precision())
        self.precision.append(self.rf.precision())
        return self.precision

    def Recall(self):
        self.recall.append(self.lr.recall())
        self.recall.append(self.dt.recall())
        self.recall.append(self.rf.recall())
        return self.recall

    def F1_score(self):
        self.f1_score.append(self.lr.f1_scores())
        self.f1_score.append(self.dt.f1_scores())
        self.f1_score.append(self.rf.f1_scores())
        return self.f1_score

    def Roc_Auc(self):
        self.roc_auc.append(self.lr.roc_auc())
        self.roc_auc.append(self.dt.roc_auc())
        self.roc_auc.append(self.rf.roc_auc())
        return self.roc_auc
