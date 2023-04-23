# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Explanations



In this GUI project a user can enter a disaster text message, which is classified among 36 categories covering requests, medical help, etc.
The positive results of these multioutput classification are shown by highlighting the corresponding categories.
The model is based on the "word to vector" model.
Hereby, each word is represented by a vector whose digitized projections are transformed into onehotencoded vectors and summarized among a text message before entering the TFIDF-Transformer.
Corresponding results are then passed to the StandardScaler before the classifier acts.
There is also another feature added, which provides information on whether the first word of a message is a verb a retweet ("RT") or not.
The classifier consists of an augmented version of the OneVsRest-classifier which uses the LogisticRegression classifier.
The model uses 80% of the data for training set and the classifier is converging to its final solution using the cross validation method GridSearchCV.


During tests when making this project, it turned out that the word 3to vector model only yields slightly better results than the bag of words approach.

Concerning the training data, one can recognize that text messages for which a relation to categories was investigated are marked with "related=1".
Non investiated text messages are categorized with 0 throughout all target variables (including "related").
