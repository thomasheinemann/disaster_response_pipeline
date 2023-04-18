# import libraries
import sys
import pandas as pd

from sqlalchemy import create_engine

#libraries for pickling
import io
try:
    import joblib
except:
    from sklearn.externals import joblib


import numpy as np

from transformer_module import tokenize, w2v
from adjusted_classifier import adjusted_classifier
from sklearn.model_selection import train_test_split


def load_data(database_filepath):
    # load data from database
    engine = create_engine("sqlite:///"+database_filepath)
    df = pd.read_sql_table('mytable', con=engine)
    df=df[[df.message[i] is not None for i in range(0, len(df))]]
    X = df.message.iloc[:]
    Y = df.iloc[:,4:]

    return X,Y, Y.columns
#####################################################




#libraries for transformations
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import MaxAbsScaler

import nltk
from sklearn.base import BaseEstimator,TransformerMixin
class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            if len(pos_tags)>0:
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)






#sparse arrays
import scipy.sparse

#classifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

def build_model():

    """Definition of the model via pipeline and a parameters variable.
       The best paramters were determined using grid search.
    """
    # pipeline_classic = Pipeline([
    #
    #     ('features', FeatureUnion([
    #
    #         ('pipeline0', Pipeline([
    #             ('countvec', CountVectorizer(tokenizer=tokenize)),
    #             ('tfidf', TfidfTransformer()),
    #             ('norm', MaxAbsScaler())
    #         ]))
    #
    #     ])),
    #
    #     ('clf',MultiOutputClassifier(estimator=adjusted_classifier(),n_jobs=-1))
    # ])
    # parameters_classic = [{
    # 'clf__estimator' : ([adjusted_classifier(LogisticRegression,0.00,1,1),adjusted_classifier(LogisticRegression,0.005,1,1)])
    # }]
    # cv = GridSearchCV(pipeline_classic,param_grid=parameters_classic, scoring='f1_macro')

    # pipeline_classic = Pipeline([
    #
    #     ('features', FeatureUnion([
    #
    #         ('pipeline0', Pipeline([
    #             ('countvec', CountVectorizer(tokenizer=tokenize)),
    #             ('tfidf', TfidfTransformer()),
    #             ('norm', MaxAbsScaler())
    #         ])),
    #
    #         ('starting_verb', StartingVerbExtractor())
    #
    #     ])),
    #
    #     ('clf',MultiOutputClassifier(estimator=adjusted_classifier(),n_jobs=-1))
    # ])
    # parameters_classic = [{
    # 'clf__estimator' : ([adjusted_classifier(LogisticRegression,0.00,1,1)])
    # }]
    # cv = GridSearchCV(pipeline_classic,param_grid=parameters_classic)# , scoring='precision')

    # pipeline_mixed = Pipeline([
    #
    #     ('features', FeatureUnion([
    #
    #         ('pipeline0', Pipeline([
    #             ('countvec', CountVectorizer(tokenizer=tokenize)),
    #             ('tfidf', TfidfTransformer()),
    #             ('norm', MaxAbsScaler())
    #         ])),
    #
    #         ('starting_verb', StartingVerbExtractor())
    #
    #     ])),
    #
    #     ('clf',MultiOutputClassifier(estimator=adjusted_classifier(),n_jobs=-1))
    # ])
    # parameters_mixed = [{
    # 'clf__estimator' : ([adjusted_classifier(LogisticRegression,0.00,1,1)])
    # }]
    # cv = GridSearchCV(pipeline_mixed,param_grid=parameters_mixed)# , scoring='precision')

    # pipeline_advanced = Pipeline([
    #
    #     ('features', FeatureUnion([
    #
    #         ('pipeline1', Pipeline([
    #             ('word2vec', w2v()),
    #             ('tfidf', TfidfTransformer()),
    #             ('norm', MaxAbsScaler())
    #         ])),
    #         #
    #         # ('pipeline2', Pipeline([
    #         #     ('word2vec', w2v()),
    #         #     ('tfidf', TfidfTransformer()),
    #         #     ('norm', MaxAbsScaler())
    #         # ])),
    #
    #         ('starting_verb', StartingVerbExtractor())
    #
    #     ])),
    #
    #     ('clf',MultiOutputClassifier(estimator=adjusted_classifier(),n_jobs=-1))
    # ])
    #
    # parameters_advanced = [{
    # 'features__pipeline1__word2vec__resolution': ([[5,5,5,5,5,5,5,4],[5,5,5,5,5,5,5,2]]),
    # 'features__pipeline1__word2vec__window' : ([20]),
    # 'features__pipeline1__word2vec__min_count' : ([1]),
    # 'features__pipeline1__word2vec__epochs' : ([50]),
    # # 'features__pipeline2__word2vec__resolution': ([[165000]]),
    # # 'features__pipeline2__word2vec__window' : ([20]),
    # # 'features__pipeline2__word2vec__min_count' : ([1]),
    # # 'features__pipeline2__word2vec__epochs' : ([50]),
    # 'clf__estimator' : ([adjusted_classifier(LogisticRegression,0.0,1,1)])
    # #'clf__estimator' : ([adjusted_classifier(LogisticRegression,0.05,1,10),adjusted_classifier(LogisticRegression,0.1,1,10),adjusted_classifier(LogisticRegression,0.02,1,10),adjusted_classifier(LogisticRegression,0.05,1,1),adjusted_classifier(LogisticRegression,0.05,0.1,1),adjusted_classifier(LogisticRegression,0.05,0.5,10)])
    # }
    # ]
    #
    #
    #
    # cv = GridSearchCV(pipeline_advanced,param_grid=parameters_advanced)#,refit=True, scoring='accuracy')



    pipeline_advanced = Pipeline([

        ('features', FeatureUnion([

            ('pipeline1', Pipeline([
                ('word2vec', w2v()),
                ('tfidf', TfidfTransformer()),
                ('norm', MaxAbsScaler())
            ])),

            ('starting_verb', StartingVerbExtractor())

        ])),

        ('clf',MultiOutputClassifier(estimator=adjusted_classifier(),n_jobs=-1))
    ])

    parameters_advanced = [{
    # 'features__pipeline1__word2vec__resolution': ([[5,5,5,5,5,5,5,3],[5,5,5,5,5,5,5,4],[5,5,5,5,5,5,5,5],[5,5,5,5,5,5,5,5,2]]),
    # 'features__pipeline1__word2vec__resolution': ([[5,5,5,5,5,5,5,5],[5,5,5,5,5,5,5,6], [5,5,5,5,5,5,5,7],[5,5,5,5,5,5,5,8],[5,5,5,5,5,5,5,9],[5,5,5,5,5,5,5,10],[5,5,5,5,5,5,5,4]]),
    # 'features__pipeline1__word2vec__resolution': ([[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],[5,5,5,5,5,5,5,8],[4,4,4,4,4,4,4,4,10]]),
    # 'features__pipeline1__word2vec__resolution': ([[6,6,6,6,6,6,6,2]]),
    'features__pipeline1__word2vec__resolution': ([[6]]),
    # 'features__pipeline1__word2vec__resolution': ([[6,6,6,6,6,6,6,2],[6,6,6,6,6,6,6,3],[5,5,5,5,5,5,5,8],[5,5,5,5,5,6,6,6],[5,5,5,5,6,6,6,6]]),
    'features__pipeline1__word2vec__window' : ([20]),
    'features__pipeline1__word2vec__min_count' : ([1]),
    'features__pipeline1__word2vec__epochs' : ([50]),
    'clf__estimator' : ([adjusted_classifier(LogisticRegression,1)])
    }
    ]



    cv = GridSearchCV(pipeline_advanced,param_grid=parameters_advanced,refit=True, scoring='f1_macro')
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """model is evaluated"""

    print("Evaluate categorical data from test set")
    y_pred=pd.DataFrame(model.predict(X_test)).T.values.tolist()
    print(Y_test.shape)
    print("\n")
    print("Evaluate Categorical estimator:")
    print("\n")

    print(model.best_params_)
    for i in range(len(category_names)):
        print(category_names[i])
        print(classification_report(Y_test.iloc[:,i],y_pred[i],zero_division=1))

#

def save_model(model, model_filepath):
    """model is pickled into a file"""

    outfile = open(model_filepath,'wb')
    joblib.dump(model,outfile,compress=7)
    outfile.close()


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X2, Y2, category_names = load_data(database_filepath)
        X=X2[0:]
        Y=Y2.iloc[0:,0:9]

        category_names=Y.columns

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train[Y_train.iloc[:,0]==1],Y_train[Y_train.iloc[:,0]==1].iloc[:,0:9])
        print('Evaluating model...')
        print(model.score(X_test[Y_test.iloc[:,0]==1],Y_test[Y_test.iloc[:,0]==1].iloc[:,0:9]))
        print(model.cv_results_['params'])
        evaluate_model(model, X_test[Y_test.iloc[:,0]==1],Y_test[Y_test.iloc[:,0]==1].iloc[:,0:9], category_names[0:9])
        # model.fit(X_train[Y_train['related']==1],Y_train[Y_train['related']=1].iloc[:,0:9])
        # print('Evaluating model...')
        # print(model.score(X_test[Y_test['related']==1],Y_test[Y_test['related']==1].iloc[:,0:9]))
        # print(model.cv_results_['params'])
        # evaluate_model(model, X_test[Y_test['related']==1],Y_test[Y_test['related']==1].iloc[:,0:9], category_names[0:9])
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
