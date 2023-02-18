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
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import MaxAbsScaler

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

    pipeline = Pipeline([
        ('countvec', CountVectorizer(tokenizer=tokenize)),
        ('word2vec', w2v()),
        ('tfidf', TfidfTransformer()),
        ('norm', MaxAbsScaler()),
        ('clf',MultiOutputClassifier(estimator=LogisticRegression(),n_jobs=-1))# hiermit klappt countvectorizer und word2vec mit [0]
        # ('clf',multiclassifier()) #hiermit klappt countvectorizer und word2vec analog mit [0]
    ])

    parameters = [{
    # MultiOutputClassifier
    # 'word2vec': ['passthrough'],
    # 'clf__estimator' : ([LogisticRegression()])
    # }
    # # MultiOutputClassifier
    # 'countvec': ['passthrough'],
    # 'word2vec__resolution': ([[60]]), #,[9,9,8,8],[7,7,7,7]]),#([[8,8,8,8],[9,8,8,8],[6,6,6,6,6]]),
    # 'word2vec__window' : ([20]),
    # 'word2vec__min_count' : ([1]),
    # 'word2vec__epochs' : ([50]),
    # }
    # 'word2vec': ['passthrough'],
    # 'clf__estimator' : ([LogisticRegression]),
    # 'clf__sample_weight_coefficient' : ([0.25]),
    # 'clf__sample_weight_threshold' : ([0.01]),
    # 'clf__C_param1' : ([1]),
    # 'clf__C_param2' : ([1])
    # }
    'countvec': ['passthrough'],
    'word2vec__resolution': ([[450,100],[45000]]),
    'word2vec__window' : ([20]),
    'word2vec__min_count' : ([1]),
    'word2vec__epochs' : ([50]),
    'clf__estimator' : ([adjusted_classifier(LogisticRegression,0.1,1000,1000)])
    }
    ]


    cv = GridSearchCV(pipeline,param_grid=parameters, scoring='f1_macro')
    # cv = pipeline
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """model is evaluated with respect to precision"""

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
