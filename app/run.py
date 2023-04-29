# basic libaries
import pandas as pd
import sys

# sql library
from sqlalchemy import create_engine

# flask library
from flask import Flask
from flask import render_template, request, jsonify

# plotly, library for diagrams
import plotly
from plotly.graph_objs import Bar, Heatmap

# json library to provide diagram configuration in json format
import json

# libraries for pickling
try:
    import joblib
except:
    from sklearn.externals import joblib

# own libraries
sys.path.append("../models")
from transformer_module import tokenize, w2v
from classifier_module import adjusted_classifier


###########
app = Flask(__name__)


# load data
engine = create_engine("sqlite:///../data/DisasterResponse.db")
df = pd.read_sql_table("mytable", engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route("/")
@app.route("/index")
def index():
    """provides flask index page with python graphics stuff"""
    # extract data needed for visuals

    df2 = df.iloc[:, 4:]
    sample_counts = df2.sum(axis=0)
    all_counts = df2.count(axis=0)[0]

    df3 = [df2[df2.iloc[:, i] == 0].sum(axis=0) for i in range(len(df2.columns))]
    df4 = [df2[df2.iloc[:, i] == 1].sum(axis=0) for i in range(len(df2.columns))]

    graphs = [
        {
            "data": [Bar(x=list(df2.columns), y=sample_counts)],
            "layout": {
                "title": "Fig. 1: Number of positive samples per category (total messages: "
                + str(all_counts)
                + ")",
                "yaxis": {"title": "Count"},
                "xaxis": {
                    "title": "Message Category",
                    "tickangle": 45,
                    "automargin": True,
                },
            },
        },
        {
            "data": [
                Heatmap(
                    z=pd.DataFrame(df3).T.values.tolist(),
                    x=df2.columns,
                    y=df2.columns,
                    hoverongaps=False,
                )
            ],
            "layout": {
                "title": "Fig. 2: Number count of category Y over all samples of training set with category X  = 0",
                "autosize": True,
                "width": 1000,
                "height": 1000,
                "xaxis": {"title": "Category X", "tickangle": 45, "automargin": True},
                "yaxis": {"title": "Category Y", "tickangle": 45, "automargin": True},
            },
        },
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template("master.html", ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route("/go")
def go():
    """provides flask 'go' page with python graphics stuff"""
    # save user input in query
    query = request.args.get("query", "")

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        "go.html", query=query, classification_result=classification_results
    )


def main():
    app.run(host="0.0.0.0", port=3001, debug=True)


if __name__ == "__main__":
    main()
