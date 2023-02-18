import sys
# import libraries
import pandas as pd
from sqlalchemy import create_engine

import nltk
nltk.download('punkt')
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
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer




#libraries for transformations

from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


class text2vec(BaseEstimator, TransformerMixin):
    model=Doc2Vec()


    def fit(self, X, y=None):

        #documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(X)]
        documents = [TaggedDocument(tokenize(doc), [i]) for i, doc in enumerate(X)]
        #print(documents)
        self.model = Doc2Vec(documents, vector_size=4, window=5, min_count=200, workers=8)
        #self.model.build_vocab(docs)
        #print("fghJ1")
        #vector = model.infer_vector(sent_tokenize('das ist sehr gut.'))
        return self

    def bla(self,X):
        #return self.model.infer_vector(sent_tokenize(X))
        #print("in bla ",type(X), len(X), ' '.join(tokenize(X)))
        #print(self.model.infer_vector(sent_tokenize(''.join(tokenize(X)))))
        self.model.random.seed(0)
        v=self.model.infer_vector(tokenize(X))
        #return v/np.linalg.norm(v)
        return v

    def transform(self, X):
        #X_tagged = pd.Series(X).apply(self.starting_verb)
        #print(self.model.infer_vector(sent_tokenize(X[1])))
        #self.model.infer_vector(sent_tokenize(X[1]))
        #return self#pd.Series(X)
        #####print( pd.Series(X))
        #print(X.head(10))
        #print( pd.Series(X).apply(self.bla).apply(pd.Series))
        #print("fghJ",pd.Series(X).shape)
        return pd.Series(X).apply(self.bla).apply(pd.Series)
    #return self #model.infer_vector(sent_tokenize('das ist sehr gut.'))# pd.DataFrame(X_tagged)




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
    #df = pd.read_sql_table('mytable', engine)
    #df=df[[df.message[i] is not None for i in range(0, len(df))]]
    #X = df.message.iloc[:]#,1:2]
    #Y = df.iloc[:,4:]


    #df = pd.read_sql("select * from mytable order by id asc", engine)
    df = pd.read_sql_table('mytable', con=engine)
    df=df[[df.message[i] is not None for i in range(0, len(df))]]
    X = df.message.iloc[:]#,1:2]
    Y = df.iloc[:,4:]

    #print(Y.head(5))
    #print(sys.version)

    return X,Y, Y.columns

def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    #text=re.sub("\d+", "number", text) #replace digits

    #text=re.sub("\"", "", text) #remove "\""

    text=re.sub("I", "", text) #remove
    text=re.sub("We", "", text) #remove

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tok=re.sub(" we ", "", clean_tok)
        clean_tok=re.sub("\"", "", clean_tok)
        clean_tok=re.sub("\d+", "number", clean_tok)



        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():


    pipeline = Pipeline([
        ('vect', text2vec()),
        #('vect', CountVectorizer(tokenizer=tokenize)),
        #('tfidf', TfidfTransformer()),
        #('clf',MultiOutputClassifier(estimator=RandomForestClassifier(random_state=0,n_estimators = 10)))#,class_weight={0:1,1:1})))#class_weight)))
        ('clf',MultiOutputClassifier(estimator=LogisticRegression(random_state=0)))#,class_weight='balanced')))#class_weight)))
        #('clf',MultiOutputClassifier(estimator=LogisticRegression(random_state=0,class_weight='balanced)))
        #('clf',MultiOutputClassifier(estimator=LogisticRegression(random_state=0,class_weight={0:1,1:1})))
        #('clf',MultiOutputClassifier(estimator=LogisticRegression(random_state=0,class_weight=class_weight)))#{0:1,1:0.3})))#,class_weight='balanced')))#class_weight)))
        #('clf',MultiOutputClassifier(estimator=LogisticRegression(random_state=0,class_weight={0:1,1:0.3})))#class_weight)))

    ])
    #quit()
    parameters = {
    # #'vect__tokenizer': (word_tokenize,tokenize),
    'clf__estimator__n_estimators': [5,5,1]
    # #'clf__estimator': (RandomForestClassifier(random_state=0), DecisionTreeClassifier(random_state=0), svm.SVC(random_state=0))
    # 'clf__estimator': (svm.SVC(),svm.SVC())
    # #'clf__estimator': (RandomForestClassifier(random_state=0), RandomForestClassifier(random_state=0))
    }


    #cv = GridSearchCV(pipeline,param_grid=parameters)
    cv = pipeline
    return cv


def evaluate_model(model, X_test, Y_test, category_names):


    from sklearn.metrics import precision_recall_fscore_support as score
    import pprint


    print("Evaluate categorical data from test set")
    y_pred=pd.DataFrame(model.predict(X_test))

    print("\n")
    print("Evaluate Categorical estimator:")
    print("\n")
    dicti={}
    for i in range(0,min(1,len(category_names)),1):

        #print("Category: ",Y_test.columns[i])
        #cr=classification_report(pd.DataFrame(Y_test.values).iloc[:,i], y_pred.iloc[:,i],labels=[0,1])
        #print(cr)
        precision,recall,fscore,support=score(pd.DataFrame(Y_test.values).iloc[:,i], y_pred.iloc[:,i],average='macro')
        #print('Precision : {}'.format(precision))
        #print('Recall    : {}'.format(recall))
        #print('F-score   : {}'.format(fscore))
        #print('Support   : {}'.format(support))
        dicti[Y_test.columns[i]]=precision
        #print(Y_test.columns[i],'\t', ': {}'.format(precision))
    print(dicti)



def save_model(model, model_filepath):


    #class MyClass:
    #    my_attribute = 1
    outfile = open(model_filepath,'wb')
    joblib.dump(model,outfile,compress=7)
    outfile.close()


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        #Y.drop(category_names[9])
        for i in range(len(Y.columns)-1,8,-1):
            Y.drop(columns=[Y.columns[i]],inplace=True)#Y.drop(columns=[Y.columns[9]],inplace=True)
        category_names=Y.columns

        #quit()
        print(Y.shape, category_names, type(category_names))
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
        #print(len(Y_train)/Y_train.iloc[:,0].sum()-1)
        #print(len(Y_train)-Y_train.iloc[:,0].sum(),len(Y_train),Y_train.iloc[:,0].sum())
        #print([{0:1,1:len(Y_train)/Y_train.iloc[:,i].sum()-1} for i in range(0, len(Y_train.columns),1)])
        #quit()
        class_weight={"output"+str(i+1) : {0:1,1:len(Y_train)/(Y_train.iloc[:,i].sum()+0.1)-1} for i in range(0, len(Y_train.columns),1)}
        print(class_weight)
        print('Building model...')
        model = build_model()
        print(X_train.shape)
        print(Y_train.shape)
        print('Training model...')

        #model.fit(X_train, Y_train.iloc[:,1:2])


        #partial first
        for i in range(0,Y_train.shape[1],1):

            f1=(Y_train.iloc[:,i].sum()/Y_train.shape[0])
            f2=1.0-f1

            sample_weight=Y_train.iloc[:,i].apply(lambda x : f2 if x==1 else f1)
            model.fit(X_train, Y_train.iloc[:,i],clf__sample_weight=sample_weight)

        #
        #quit()
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test.iloc[:,1:2], category_names)
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
