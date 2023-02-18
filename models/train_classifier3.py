import sys
# import libraries
import pandas as pd
from sqlalchemy import create_engine

import nltk
#nltk.download('punkt')
from nltk.tokenize import word_tokenize

from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
#
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

#classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV


#Libraries for tokenization
import nltk
#nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

#libraries for transformations

from sklearn.base import BaseEstimator, TransformerMixin
#from gensim.models.doc2vec import Doc2Vec, TaggedDocument

#scrambling

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.naive_bayes import CategoricalNB


class scaler(TransformerMixin):
    def __init__(self):
        pass


    def fit(self, X, y=None):

        #documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(X)]
        documents = [TaggedDocument(tokenize(doc), [i]) for i, doc in enumerate(X)]
        #print(documents)
        self.model = Doc2Vec(documents, vector_size=2000, window=200, min_count=20, workers=8)
        #self.model.build_vocab(docs)
        #print("fghJ1")
        #vector = model.infer_vector(sent_tokenize('das ist sehr gut.'))
        return self

    def infer(self,X):
        v=X-np.mean(X)

        return v

    def transform(self, X):   #for test data

        return pd.Series(X).apply(self.infer).apply(pd.Series)
    #return self #model.infer_vector(sent_tokenize('das ist sehr gut.'))# pd.DataFrame(X_tagged)

class multiclassifier(BaseEstimator, TransformerMixin):
    #model=MultiOutputClassifier(LogisticRegression())
    def __init__(self, estimator, weights=None):
        self.estimator = estimator
        #self.estimator.class_weight='balanced'
        self.weights = weights
        self.targets=1

    def fit(self, X, y=None):

        self.model=[self.estimator for i in range(y.shape[1])]
        #self.model.fit(X,y)
        self.targets=y.shape[1]
        for i in range(self.targets):
            #print(i)
            f1=(y.iloc[:,i].sum()/y.shape[0])
            f2=1.0-f1
            if f1 ==0:
                y.iloc[0,i]=1
            if f2==0:
                y.iloc[0,i]=0
            f1=(y.iloc[:,i].sum()/y.shape[0])
            f2=1.0-f1
            self.estimator.class_weight={0:f2,1:f1}

            self.model[i].fit(X,y.iloc[:,i]) #,sample_weight=sample_weight)#,classes=[np.array([0,1])])##,sample_weight=sample_weight)
        return self
    def predict(self,X):
        y_pred=[]
        #print("X-shape: ",X.shape)
        for i in range(self.targets):
            if i==0:
                y_pred=pd.DataFrame(self.model[i].predict(X))
            else:
                y_pred[i]=self.model[i].predict(X)
            print(self.model[i].predict(X).shape)

        return y_pred




#libraries for pickling
import io
try:
    import joblib
except:
    from sklearn.externals import joblib as joblib
#tetlibs
from nltk.tokenize import sent_tokenize

def load_data(database_filepath):
    # load data from database
    engine = create_engine("sqlite:///"+database_filepath)
    df = pd.read_sql_table('mytable', con=engine)
    df=df[[df.message[i] is not None for i in range(0, len(df))]]
    X = df.message.iloc[:]
    Y = df.iloc[:,4:]

    return X,Y, Y.columns

def tokenize(text):
    """
        Tokenize text and replace URLs with a placeholder and delete non-specific words
        output: Array-like tokens
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")
    clean_tokens = []

    for tok in tokens:
        if tok not in stop_words:
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()

            if clean_tok not in list(["we", "the", "a", "an", "\""]):
                clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():


    pipeline = Pipeline([
        #('vect', text2vec()),
        #('scl',StandardScaler()),
        #('scale',MinMaxScaler()),
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        #('scale',MinMaxScaler()),
        #('scale',MaxAbsScaler()),
        #('scl',StandardScaler(copy=False,with_mean=True)),
        #('clf',MultiOutputClassifier(estimator=SVC(random_state=0)))#,class_weight={0:1,1:1})))#class_weight)))
        ('clf',MultiOutputClassifier(estimator=SVC()))
        #('clf',multiclassifier(SVC()))#class_weight)))
        #('clf',MultiOutputClassifier(estimator=LogisticRegression(random_state=0,max_iter=1000)))#,class_weight='balanced')))

    ])
    #quit()
    parameters = {
    # # #'vect__tokenizer': (word_tokenize,tokenize),
    # 'clf__estimator__n_estimators': [5,5,1]
    # # #'clf__estimator': (RandomForestClassifier(random_state=0), DecisionTreeClassifier(random_state=0), svm.SVC(random_state=0))
    # # 'clf__estimator': (svm.SVC(),svm.SVC())
    # # #'clf__estimator': (RandomForestClassifier(random_state=0), RandomForestClassifier(random_state=0))
    }


    #cv = GridSearchCV(pipeline,param_grid=parameters)
    cv = pipeline
    return cv


def evaluate_model(model, X_test, Y_test, category_names):


    from sklearn.metrics import precision_recall_fscore_support as score
    import pprint


    print("Evaluate categorical data from test set")
    y_pred=pd.DataFrame(model.predict(X_test))
    #print(y_pred.head(5))
    #print(y_pred.shape)
    print("\n")
    print("Evaluate Categorical estimator:")
    print("\n")
    dicti={}
    for i in range(0,min(8,len(category_names)),1):

        #print("Category: ",Y_test.columns[i])
        #cr=classification_report(pd.DataFrame(Y_test.values).iloc[:,i], y_pred.iloc[:,i],labels=[0,1])
        #print(cr)
        #precision,recall,fscore,support=score(pd.DataFrame(Y_test.values).iloc[:,i], y_pred.iloc[:,i],average='macro',zero_division=0)
        precision,recall,fscore,support=score(pd.DataFrame(Y_test.values).iloc[:,i], y_pred.iloc[:,i],average='macro',zero_division=0)
        #print('Precision : {}'.format(precision))
        #print('Recall    : {}'.format(recall))
        #print('F-score   : {}'.format(fscore))
        #print('Support   : {}'.format(support))
        dicti[Y_test.columns[i]]=precision
        #print(Y_test.columns[i],'\t', ': {}'.format(precision))
    print(dicti)

#

def save_model(model, model_filepath):
    outfile = open(model_filepath,'wb')
    joblib.dump(model,outfile,compress=7)
    outfile.close()


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)

        #for i in range(len(Y.columns)-1,8,-1):
        #    Y.drop(columns=[Y.columns[i]],inplace=True)#Y.drop(columns=[Y.columns[9]],inplace=True)
        category_names=Y.columns

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

        #class_weight={"output"+str(i+1) : {0:1,1:len(Y_train)/(Y_train.iloc[:,i].sum()+0.1)-1} for i in range(0, len(Y_train.columns),1)}
        #print(class_weight)
        print('Building model...')
        model = build_model()
        #print(X_train.shape)
        #print(Y_train.shape)
        print('Training model...')

        model.fit(X_train, Y_train.iloc[:,0:8])



        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)
        #print('beste Parameter:',model.best_params_)
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
