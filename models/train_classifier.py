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
#from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

#classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

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


def load_data(database_filepath):
    # load data from database
    engine = create_engine("sqlite:///"+database_filepath)
    df = pd.read_sql_table('mytable', engine)
    df=df[[df.message[i] is not None for i in range(0, len(df))]]
    X = df.message.iloc[:]#,1:2]
    Y = df.iloc[:,4:]

    return X,Y, Y.columns

def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf',MultiOutputClassifier(estimator=RandomForestClassifier(random_state=0,n_estimators = 20)))
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    print("Evaluate categorical data from test set")
    y_pred=pd.DataFrame(model.predict(X_test))

    print("\n")
    print("Evaluate Categorical estimator:")
    print("\n")
    for i in range(0,10,1):
        print("Category: ",Y_test.columns[i])
        print(classification_report(pd.DataFrame(Y_test.values).iloc[:,i], y_pred.iloc[:,i],labels=[0,1]))#, target_names=target_names))


def save_model(model, model_filepath):
    import io
    import joblib

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
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()
        print(X_train.shape)
        print(Y_train.shape)
        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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
