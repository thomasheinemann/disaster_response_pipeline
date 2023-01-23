import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
#from sklearn.externals import joblib
import joblib





###########
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

class text2vec(BaseEstimator, TransformerMixin):
    model=Doc2Vec()


    def fit(self, X, y=None):

        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(X)]
        #print(documents)
        self.model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)

        #vector = model.infer_vector(sent_tokenize('das ist sehr gut.'))
        return self
    def bla(self,X):
        return self.model.infer_vector(sent_tokenize(X))

    def transform(self, X):
        #X_tagged = pd.Series(X).apply(self.starting_verb)
        #print(self.model.infer_vector(sent_tokenize(X[1])))
        #self.model.infer_vector(sent_tokenize(X[1]))
        #return self#pd.Series(X)
        #####print( pd.Series(X))
        #print(X.head(10))
        #print( pd.Series(X).apply(self.bla).apply(pd.Series))
        return pd.Series(X).apply(self.bla).apply(pd.Series)
    #return self #model.infer_vector(sent_tokenize('das ist sehr gut.'))# pd.DataFrame(X_tagged)
#########






from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('mytable', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
