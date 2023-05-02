# Disaster Response Pipeline Project

![cars](cars.png)
<!--
Pictures taken from:
* https://www.freepik.com/premium-vector/fire-truck-cartoon-clipart-colored-illustration_24434623.htm
* https://de.freepik.com/vektoren-premium/polizeiauto-cartoon-clipart-farbige-illustration_24434932.htm
* https://cobi.eu/product/barkas-b1000-krankenwagen,3403
-->
### Summary

This disaster response project is a web app triggered by the udacity "data-scientist-nanodegree" program (see https://www.udacity.com/).
Hereby, a user can enter into a textbox a disaster message, which is then classified among 36 categories covering requests, medical help, etc.
The positive results of these multioutput classifications are shown by highlighting the corresponding categories.


### Use case / motivation
If an emergency message is received by an emergency service, the operator has to decide quickly what to do.
The callers are often in shock or panicked - so no information should get lost.
In such moments, time is valuable and seconds may decide over life or death.
The following exemplary questions may arise: What emergency services are required? How many emergency cars of each service are needed? How many more are to be held on standby? Are special technical machineries like helicopters or heavy fireworks cars in need? Could the situation escalate? ...
A first help would be to understand the kind of emergency. Often it is even more than one.
Given this problem, this app could be used in supporting the operator by detecting emergency categories such that he or she could make a quick but reasonable decision.
The requirement is that the message is transcripted.



### Raw model / fields of application

This project serves as a simple raw model for a multi-target classification of text messages. It consists of fundamental natural language processing steps
and possesses a machine-learning pipeline.
There are various fields of direct application.
Given a text, one may analyze its tonality or topic, the language, the dialect, or even diagnostic findings, etc.





### Install/run instructions:
0. Install the packages denoted in requirements_working_configuration.txt preferably in a virtual environment as exemplarily shown for the windows command prompt:
```
      projectfolder:> python -m venv venv
      projectfolder:> cd venv\Scripts
      projectfolder\venv\Scripts> .\activate.bat
      projectfolder\venv\Scripts> cd ..\..
      projectfolder:> python -m pip install -r requirements_working_configuration.txt
```
      Within your project folder "projectfolder" use the "python" command as long as the virtual environment is activated
      (if not working with/on the project, the virtual environment should be deactivated by executing projectfolder\venv\Scripts\deactivate.bat).

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
        (an exemplary output for an 80-20 train-test split is provided in ml_pipeline_output_example.txt (corresponding pickle file is in models/classifier.pkl - a recreation takes several hours))

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to the displayed address - e.g., http://192.168.1.3:3001/


### Files in the repository
```
app            # contains files for running web app
| - template
| |- master.html                       # main page of web app
| |- go.html                           # classification result page of web app
|- run.py                              # Flask file that runs app
css            # style sheet folder
data           # contains files covering ETL pipeline
|- disaster_categories.csv             # data to process
|- disaster_messages.csv               # data to process
|- process_data.py                     # program for processing data and safe into following database file
|- InsertDatabaseName.db               # database to save clean data to
models         # contains files for building the ML model
|- transformer_module.py               # module containing custom-made transformer and tokenizer
|- train_classifier.py                 # program containg ML pipeline
|- classifier_module.py                # module containing custom-made classifier
|- classifier.pkl                      # saved model
cars.png                               # image file containing cars - used in text intro
ml_pipeline_output_example.txt         # output of ML pipeline
README.md
requirements.txt                       # list of required packages
requirements_working_configuration.txt # packages covering working configuration
use_case_example.png                   # typical use case picture
.gitignore
```

### Technical explanations


The model is based on the "word to vector" (word2vec) model (https://github.com/RaRe-Technologies/gensim) in which contrary to the more simple bag of words approach, the embedding to nearby words is effectively taken into account.
In the framework of this model, each word is represented by a vector.
The higher the dimensionality of these vectors, the more expressive the Euclidian distances between words for representing the meaning of each word.
Similar words have thus a very short distance.
As the intention is to group similar-meaning words, one may not increase the dimensionality too much; otherwise, all words would have their distinct position in a high dimensional vector space and would stand out uniquely (as in the bag of words approach).
So, one has to compromise/optimize.

In the fit procedure, the digitized projections of these vectors are transformed into one-hot encoded vectors.
For each text message, a histogram of all one-hot encoded vectors is created. The plethora of histograms then enters the TF-IDF-Transformer.
Corresponding results are then passed to the StandardScaler (its mean stays untouched due to the sparsity of data) and finally passed to the classifier.
The classifier consists of an augmented version of the OneVsRest-classifier which uses the LogisticRegression classifier.

The classifier is converging to its final solution using the cross-validation method GridSearchCV with the f1-score of class 1 forming the metric that is aimed to be maximized.
The f1-score of class 1 was chosen over accuracy since one prefers in emergencies rather having a false positive classification (affecting the precision of class 1) than a false negative (affecting precision and recall of class 1).
This strategy together with a balanced class weight pushes the sensitivity of the classifier in the right direction as in the majority of cases a positive classification is rare and a classifier that always produces "0" would otherwise produce a reasonable accuracy.

To prove that the word2vec model can be superior to the bag of words model, the hyperparameter setting covers one setting for the word2vec and one for the bag of words approach.
GridSearchCV-algorithm checks out both settings and compares so-found scores.
It turns out that the set comprising the word2vec model wins.
The difference in score is about 0.358 vs 0.345 for an 80-20 train-test-split.

Concerning the training data is worth mentioning that one can recognize that text messages, for which a relation to categories was investigated, are categorized with "related=1".
Non-investigated text messages are categorized with "0" throughout all target variables (including "related" - see Figure 2 in the web app).
These non-investigated text messages were excluded from training and evaluation as one should not know from the text alone whether it was investigated or not.

In the following, a graphical representation of the disaster message "I need medicine" is shown.
![use_case_example](use_case_example.png)
