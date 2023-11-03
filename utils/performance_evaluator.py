import os

import pandas as pd
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, confusion_matrix
from sklearn.metrics import matthews_corrcoef,f1_score


class PerformanceEvaluator(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.label= []
        self.prob = []
        self.pred = []
        self.total = 0
        self.loss = []

    def update(self, label, predict, **kwargs):
        self.total += len(label)
        self.label.extend(label)
        self.prob.extend(predict)
        if 'loss' in kwargs:
            self.loss.append(kwargs['loss'])

        for item in predict:
            if item >= 0.5:
                self.pred.append(1)
            else:
                self.pred.append(0)
        

    def cal_performance(self, save_dir=None):
        
        cor = 0
        for index in range(self.total):
            if self.label[index] == self.pred[index]:
                cor += 1
                
        self.accu = cor / self.total
        self.auc = roc_auc_score(self.label, self.prob)
        precision, recall, _thresholds = precision_recall_curve(self.label, self.prob)
        self.prauc = auc(recall, precision)
        self.MCC = matthews_corrcoef(self.label, self.pred)
        self.F1_score = f1_score(self.label, self.pred)
        if self.loss:
            self.loss_avg = np.mean(self.loss)
        # sensitivity
        tn, fp, fn, tp = confusion_matrix(self.label, self.pred).ravel()
        self.sensitivity = tp / (tp + fn)
        self.specificity = tn / (tn + fp)

        # top-k recall
        self.result = [x for _, x in sorted(zip(self.prob, self.label), reverse=True)]
        self.top10_hit =  sum(self.result[0:10])
        self.top20_hit =  sum(self.result[0:20])
        self.top50_hit =  sum(self.result[0:50])



        # if len(self.prob) > 1000:
        #     self.top50_accu =0
        #     self.top50_index = sorted(range(len(self.prob)), key=lambda i: self.prob[i])[-50:]
        #     for idx in self.top50_index:
        #         if self.label[idx]:
        #             self.top50_accu += 1



        # if save_dir:
        #     data_dict = {'accu' : [self.accu], "auc" : [self.auc],  "prauc": [self.prauc]}
        #     data = pd.DataFrame.from_dict(data_dict)
        #     data.to_csv(os.path.join(save_dir, 'metric.csv'))

        
