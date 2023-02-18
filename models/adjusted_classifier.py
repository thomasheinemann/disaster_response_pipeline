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
#postprocessing
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import MaxAbsScaler
#from sklearn.naive_bayes import CategoricalNB


from sklearn.model_selection import train_test_split

#sparse arrays
import scipy.sparse

from scipy.sparse import csr_matrix,vstack, hstack

from sklearn.base import BaseEstimator
class adjusted_classifier(BaseEstimator):
    def __init__(self, estimator=LogisticRegression,sample_weight_threshold=0.1,C_param1=100,C_param2=100):
        self.set_params(estimator,sample_weight_threshold,C_param1,C_param2)
        return None
    def set_params(self,estimator,sample_weight_threshold,C_param1,C_param2):
        self.estimator = estimator
        self.sample_weight_threshold=sample_weight_threshold
        self.C_param1=C_param1
        self.C_param2=C_param2
        self.classes_=[0,1]

        return self
    def score(self,X,y):
        """scoring function being an average value over all target accuracies"""
        return self.model.score(X,y)

    def fit(self, X, y=None):
        """fit procedure covering all classifiers for each target"""
        y2=y.copy()# [y.iloc[:,0]==1]
        X2=X.copy()# [y.iloc[:,0]==1]

        f1=(y2.sum()/y2.shape[0])
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


        if f1>self.sample_weight_threshold and 1.0-f1>self.sample_weight_threshold:
            self.model=OneVsRestClassifier(self.estimator(random_state=0,C=self.C_param1,max_iter=5000))# ,class_weight='balanced'))
            # self.model2=self.estimator(random_state=0,C=self.C_param1,max_iter=5000,class_weight='balanced')
            # self.model[i]=self.estimator(random_state=0,C=1000)
            # self.model[i]=self.estimator(C=1000)
            # self.model[i]=self.estimator()
        else:
            self.model=OneVsRestClassifier(self.estimator(random_state=0,C=self.C_param2,max_iter=5000,class_weight='balanced'))
            # self.model2=self.estimator(random_state=0,C=self.C_param2,max_iter=5000,class_weight='balanced')
            # self.model[i]=self.estimator(random_state=0,C=1000)
            # self.model[i]=self.estimator(C=1000)
            # self.model[i]=self.estimator()

        self.model.fit(X2,y2)
        return self
        # if f1>self.sample_weight_threshold and 1.0-f1>self.sample_weight_threshold:
        #     self.model=self.estimator(random_state=0,C=self.C_param1,max_iter=5000)
        #     # self.model2=self.estimator(random_state=0,C=self.C_param1,max_iter=5000,class_weight='balanced')
        #     # self.model[i]=self.estimator(random_state=0,C=1000)
        #     # self.model[i]=self.estimator(C=1000)
        #     # self.model[i]=self.estimator()
        # else:
        #     self.model=self.estimator(random_state=0,C=self.C_param2,max_iter=5000)
        #     # self.model2=self.estimator(random_state=0,C=self.C_param2,max_iter=5000,class_weight='balanced')
        #     # self.model[i]=self.estimator(random_state=0,C=1000)
        #     # self.model[i]=self.estimator(C=1000)
        #     # self.model[i]=self.estimator()
        # self.model2=OneVsRestClassifier(LogisticRegression(random_state=0,C=self.C_param2,max_iter=5000,class_weight='balanced'))
        # self.model.fit(X2,y2)
        # y_pred=self.model.predict(X2)
        # X3=hstack((X2,np.array(y_pred)[:,None]))
        # self.model2.fit(X3,y2)
        #
        # return self
    def predict(self,X):
        """predict funtion being a comnanated list for predictions of each target value"""
        if self.one_class_only:
            return [self.one_class_value for i in range(X.shape[0])]
        y_pred=self.model.predict(X)
        # X3=hstack((X,np.array(y_pred)[:,None]))
        # y_pred=self.model2.predict(X3)
        return y_pred
