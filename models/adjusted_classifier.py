import numpy as np
import pandas as pd

#classifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain

from sklearn.model_selection import train_test_split

#sparse arrays
import scipy.sparse

from scipy.sparse import csr_matrix,vstack, hstack

from sklearn.base import BaseEstimator
class adjusted_classifier(BaseEstimator):
    def __init__(self, estimator=LogisticRegression,C_param=100):
        self.set_params(estimator,C_param)
        return None
    def set_params(self,estimator,C_param):
        self.estimator = estimator
        self.C_param=C_param
        self.classes_=[0,1]

        return self
    def score(self,X,y):
        """scoring function being an average value over all target accuracies"""
        print("score")
        return self.model.score(X,y)

    def fit(self, X, y=None):
        """fit procedure covering all classifiers for each target"""

        f1=(y.sum()/y.shape[0])
        if f1==0:
            self.one_class_only=True
            self.one_class_value=0
            return self
        elif f1==1:
            self.one_class_only=True
            self.one_class_value=1
            return self
        else:
            self.one_class_only=False

        self.model=OneVsRestClassifier(self.estimator(random_state=0,C=self.C_param,max_iter=5000,class_weight='balanced'))
        self.model.fit(X,y)
        return self
    def predict(self,X):
        """predict funtion for the only one target value"""
        if self.one_class_only:
            return [self.one_class_value for i in range(X.shape[0])]
        y_pred=self.model.predict(X)
        return y_pred
