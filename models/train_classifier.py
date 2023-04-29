# basic libraries
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import pprint

# library for handling sparse arrays
import scipy.sparse

# library for DB server
from sqlalchemy import create_engine

# libraries for pickling
try:
    import joblib
except:
    from sklearn.externals import joblib

# train test split library
from sklearn.model_selection import train_test_split

# transformer libraries
from transformer_module import w2v, tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import StandardScaler

# classifier libraries
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from classifier_module import adjusted_classifier

# ML pipeline library
from sklearn.pipeline import Pipeline

# hyperparameter search library
from sklearn.model_selection import GridSearchCV

# scoring libraries
from sklearn.metrics import make_scorer, f1_score, classification_report


def date_diff_in_seconds(dt2, dt1):  # from https://www.w3resource.com/python-exercises/date-time-exercise/python-date-time-exercise-37.php
    """calculates difference in seconds between two "datetime" objects"""
    timedelta = dt2 - dt1
    return timedelta.days * 24 * 3600 + timedelta.seconds


def dhms_from_seconds(seconds):  # from https://www.w3resource.com/python-exercises/date-time-exercise/python-date-time-exercise-37.php
    """transforms seconds into dhms format"""
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    return (days, hours, minutes, seconds)


def load_data(database_filepath):
    """loada data from database"""
    engine = create_engine("sqlite:///" + database_filepath)
    df = pd.read_sql_table("mytable", con=engine)
    df = df[[df.message[i] is not None for i in range(0, len(df))]]
    X = df.message.iloc[:]
    Y = df.iloc[:, 4:]

    return X, Y, Y.columns


def build_model():
    """Definition of the model via pipeline and a parameters variable.
    The best parameters were determined using grid search.
    In this example, the bag of words approach and the word-to-vetor approach are
    """

    pipeline = Pipeline(
        [
            (
                "countvec",
                CountVectorizer(tokenizer=tokenize, token_pattern=None),
            ),
            ("word2vec", w2v()),
            ("tfidf", TfidfTransformer()),
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", MultiOutputClassifier(estimator=adjusted_classifier(), n_jobs=-1))
        ]
    )

    parameters = [
        {
            "countvec": ["passthrough"],
            "word2vec__resolution":  ([[5,5,5,5,5,6,6,6]]), # ([[5,5,5,5,5,6,6,6],[5,5,5,5,5,5,5,8]]),
            "word2vec__window": ([20]),
            "word2vec__min_count": ([1]),
            "word2vec__epochs": ([50]),
            "clf__estimator": ([adjusted_classifier(LogisticRegression, 1)])
        },
        {
            "word2vec": ["passthrough"],
            "clf__estimator": ([adjusted_classifier(LogisticRegression, 1)])
        },
    ]

    cv = GridSearchCV(
        pipeline,
        param_grid=parameters,
        refit=True,
        scoring=make_scorer(f1_score, **dict(average="macro", pos_label=1, zero_division=0))
    )
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """model is evaluated as a whole and for each target variable"""

    print(
        "\nTarget-averaged f1-score for class=1:\n    ",
        model.score(
            X_test,
            Y_test
        )
    )
    y_pred = pd.DataFrame(model.predict(X_test)).T.values.tolist()
    print("\n")
    print("Evaluate Categorical estimator for each target variable:")
    print("\n")
    for i in range(len(category_names)):
        print("-----------------------------------------------------")
        print(category_names[i])
        print(classification_report(Y_test.iloc[:, i], y_pred[i], zero_division=0))


def save_model(model, model_filepath):
    """model is pickled into a file"""

    outfile = open(model_filepath, "wb")
    joblib.dump(model, outfile, compress=7)
    outfile.close()


def main():
    """procedure covering all steps of the ML-pipline"""
    if len(sys.argv) == 3:

        time1 = datetime.now()

        database_filepath, model_filepath = sys.argv[1:]
        print("Loading data...\n    DATABASE: {}".format(database_filepath))
        X, Y, category_names = load_data(database_filepath)

        category_names = Y.columns
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=0
        )

        print("\nBuilding model...")
        model = build_model()

        print("\nTraining model...")
        model.fit(
            X_train[Y_train["related"] == 1],
            Y_train[Y_train["related"] == 1].iloc[:, 0:]
        )
        print("\nRuntime for building and training the model:")
        time2 = datetime.now()
        print(
            "    %d days, %d hours, %d minutes, %d seconds"
            % dhms_from_seconds(date_diff_in_seconds(time2, time1))
        )

        print("\nEvaluating model...")

        print("\nCross validation results:")
        pprint.PrettyPrinter(indent=4).pprint(model.cv_results_)

        print("\nParameters of best estimator:")
        pprint.PrettyPrinter(indent=4).pprint(model.best_params_)

        evaluate_model(
            model,
            X_test[Y_test["related"] == 1],
            Y_test[Y_test["related"] == 1].iloc[:, 0:],
            category_names[0:]
        )

        print("\nSaving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)
        print("\nTrained model saved!")

    else:
        print(
            "Please provide the filepath of the disaster messages database "
            "as the first argument and the filepath of the pickle file to "
            "save the model to as the second argument. \n\nExample: python "
            "train_classifier.py ../data/DisasterResponse.db classifier.pkl"
        )


if __name__ == "__main__":
    main()
