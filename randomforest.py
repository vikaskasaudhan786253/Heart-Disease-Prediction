import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
with open('model_rf.pkl','rb') as file:
    model = pickle.load(file)
df = pd.read_csv('train_cleaned.csv')
x = df.iloc[:,:-1]
y = df.iloc[:,-1]
pred = model.predict(x)
proba = model.predict_proba(x)[:,1]

class Random_Forest:
    def __init__(self):
        self.model = model
        self.x = x
        self.y = y
        self.pred = pred
        self.proba = proba

    def predict(self,values):
        y_pred = model.predict(values)
        return y_pred[0]

    def predict_proba(self,values):
        y_proba = model.predict_proba(values)[:,1]
        return np.round((y_proba[0]*100),3)

    def accuracy(self):
        accu = accuracy_score(self.y,self.pred)
        return np.round((accu*100),2)

    def precision(self):
        prec = precision_score(self.y,self.pred)
        return np.round((prec*100),2)

    def recall(self):
        recall = recall_score(self.y,self.pred)
        return np.round((recall*100),2)

    def f1_scores(self):
        f1 = f1_score(self.y,self.pred)
        return np.round((f1*100),2)

    def roc_auc(self):
        roc = roc_auc_score(self.y,self.proba)
        return np.round((roc*100),2)
