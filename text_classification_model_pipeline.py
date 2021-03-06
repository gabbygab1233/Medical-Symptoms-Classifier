import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import pickle
import string
import spacy
import nltk 
import re

from sklearn.naive_bayes import *
from sklearn.ensemble import *
from sklearn.neighbors import *
from sklearn.tree import *
from sklearn.calibration import *
from sklearn.linear_model import *
from sklearn.multiclass import *
from sklearn.svm import *
from nltk.stem import WordNetLemmatizer 


from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, auc, roc_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV

from sklearn.pipeline import Pipeline, make_pipeline

#nltk.download('stopwords') 
#nltk.download('wordnet')
sns.set(style='whitegrid')

warnings.filterwarnings('ignore')

################################################################################
#                                                                              #
#                            Analyze Data                                      #
#                                                                              #
################################################################################
def explore_data(df):
    print("Number of Instances and Attributes:", df.shape)
    print('\n')
    print('Dataset columns:',df.columns)
    print('\n')
    print('Data types of each columns: ', df.info())
################################################################################
#                                                                              #
#                      Checking for Duplicates                                 #
#                                                                              #
################################################################################
def checking_removing_duplicates(df):
    count_dups = df.duplicated().sum()
    print("Number of Duplicates: ", count_dups)
    if count_dups >= 1:
        df.drop_duplicates(inplace=True)
        print('Duplicate values removed!')
    else:
        print('No Duplicate values')
################################################################################
#                                                                              #
#                     Convert Objects to Numeric                               #
#                                                                              #
################################################################################
def label_encoder(data,obj):
    le = LabelEncoder()
    le.fit(df[obj])
    data['target_encoded'] = le.transform(df[obj])
################################################################################
#                                                                              #
#                         Clean Text Data                                      #
#                                                                              #
################################################################################
def clean_txt(docs):
    lemmatizer = WordNetLemmatizer() 
    # Correct mispelled words
    #spell = SpellChecker()
    #spellcorrection = [spell.correction(x) for x in docs]
    # split into words
    speech_words = nltk.word_tokenize(docs)
    # convert to lower case
    lower_text = [w.lower() for w in speech_words]
    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    stripped = [re_punc.sub('', w) for w in lower_text]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    words = [w for w in words if not w in  list(STOP_WORDS)]
    #Stemm all the words in the sentence
    lem_words = [lemmatizer.lemmatize(word) for word in words]
    combined_text = ' '.join(lem_words)
    return combined_text
################################################################################
#                                                                              #
#                Split Data to Training and Validation set                     #
#                                                                              #
################################################################################
def read_in_and_split_data(data, features,target):
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test

################################################################################
#                                                                              #
#                        Spot-Check Algorithms                                 #
#                                                                              #
################################################################################
def Text_Model():
    Models = []
    Models.append(('MultiNB', MultinomialNB()))
    Models.append(('RF'  , RandomForestClassifier()))
    Models.append(('AdaB'  , AdaBoostClassifier()))
    Models.append(('Bagging' , BaggingClassifier()))
    Models.append(('ET'   , ExtraTreesClassifier()))
    Models.append(('GBC'  , GradientBoostingClassifier()))
    Models.append(('Cart', DecisionTreeClassifier()))
    Models.append(('CCCV', CalibratedClassifierCV()))
    Models.append(('PAC', PassiveAggressiveClassifier()))
    Models.append(('RC', RidgeClassifier()))
    Models.append(('SGD', SGDClassifier()))
    Models.append(('OVRC', OneVsRestClassifier(LogisticRegression())))
    Models.append(('KNN', KNeighborsClassifier()))
            
    return Models

################################################################################
#                                                                              #
#                 Spot-Check Normalized Text Models                            #
#                                                                              #
################################################################################
def NormalizedTextModel(nameOfvect):
    
    if nameOfvect == 'countvect':
        vectorizer = CountVectorizer()
    elif nameOfvect =='tfvect':
        vectorizer = TfidfVectorizer()
    elif nameOfvect == 'hashvect':
        vectorizer = HashingVectorizer()

    pipelines = []
    pipelines.append((nameOfvect+'MultinomialNB'  , Pipeline([('Vectorizer', vectorizer),('LR'  , LogisticRegression())])))
    pipelines.append((nameOfvect+'CCCV' , Pipeline([('Vectorizer', vectorizer),('CCCV' , CalibratedClassifierCV())])))
    pipelines.append((nameOfvect+'KNN' , Pipeline([('Vectorizer', vectorizer),('KNN' , KNeighborsClassifier())])))
    pipelines.append((nameOfvect+'CART', Pipeline([('Vectorizer', vectorizer),('CART', DecisionTreeClassifier())])))
    pipelines.append((nameOfvect+'PAC'  , Pipeline([('Vectorizer', vectorizer),('PAC'  , PassiveAggressiveClassifier())])))
    pipelines.append((nameOfvect+'SVM' , Pipeline([('Vectorizer', vectorizer),('RC' , RidgeClassifier())])))
    pipelines.append((nameOfvect+'AB'  , Pipeline([('Vectorizer', vectorizer),('AB'  , AdaBoostClassifier())])  ))
    pipelines.append((nameOfvect+'GBM' , Pipeline([('Vectorizer', vectorizer),('GMB' , GradientBoostingClassifier())])))
    pipelines.append((nameOfvect+'RF'  , Pipeline([('Vectorizer', vectorizer),('RF'  , RandomForestClassifier())])))
    pipelines.append((nameOfvect+'ET'  , Pipeline([('Vectorizer', vectorizer),('ET'  , ExtraTreesClassifier())])))
    pipelines.append((nameOfvect+'SGD'  , Pipeline([('Vectorizer', vectorizer),('SGD'  , SGDClassifier())])))
    pipelines.append((nameOfvect+'OVRC'  , Pipeline([('Vectorizer', vectorizer),('OVRC'  , OneVsRestClassifier(LogisticRegression()))])))
    pipelines.append((nameOfvect+'Bagging'  , Pipeline([('Vectorizer', vectorizer),('Bagging'  , BaggingClassifier())])))

    return pipelines
################################################################################
#                                                                              #
#                           Train Model                                        #
#                                                                              #
################################################################################
def fit_model(X_train, y_train,models):
    # Test options and evaluation metric
    num_folds = 10
    scoring = 'accuracy'

    results = []
    names = []
    for name, model in models:
        kfold = KFold(n_splits=num_folds)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
################################################################################
#                                                                              #
#                          save Trained Model                                  #
#                                                                              #
################################################################################
def save_model(model,filename):
    pickle.dump(model, open(filename, 'wb'))
################################################################################
#                                                                              #
#                          Performance Measure                                 #
#                                                                              #
################################################################################
def classification_metrics(model, y_test, y_pred):
    print(f"Training Accuracy Score: {model.score(X_train, y_train) * 100:.1f}%")
    print(f"Validation Accuracy Score: {model.score(X_test, y_test) * 100:.1f}%")
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    fig,ax = plt.subplots(figsize=(8,6))
    sns.heatmap(pd.DataFrame(conf_matrix), annot = True, cmap = 'YlGnBu',fmt = 'g')
    ax.xaxis.set_label_position('top')
    plt.tight_layout()
    plt.title('Confusion matrix for Logisitic Regression Model', fontsize=20, y=1.1)
    plt.ylabel('Actual label', fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)
    plt.show()
    print(classification_report(y_test, y_pred))
    
# Load data
df = pd.read_csv('overview-of-recordings.csv')
df_text = df[['phrase', 'prompt']]

# Cleaning the text data
df_text['phrase'] = df_text['phrase'].apply(clean_txt)

#label_encoder(df_text, 'prompt')
# Split data to training and validation set
X = 'phrase'
target_class = 'prompt'
X_train, X_test, y_train, y_test = read_in_and_split_data(df_text, X, target_class)


# Fit data to model using countvectorizer
print('\nCount Vectorizer')
models = NormalizedTextModel('countvect')
fit_model(X_train, y_train, models)


# Fit data to model using tfvectorizer
print('\nTfidfVectorizer')
models = NormalizedTextModel('tfvect')
fit_model(X_train, y_train, models)


# Fine Tuning
print('\nFine Tuning')
vectorizer = TfidfVectorizer()
X_train_1 = vectorizer.fit_transform(X_train)
model = BaggingClassifier()
n_estimators = [10, 100, 1000]
grid = dict(n_estimators=n_estimators)
cv = KFold(n_splits=10)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy')
grid_result = grid_search.fit(X_train_1, y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# Predict unseen data
print('\nPredict Unseen Data')
text_clf = Pipeline([('vect', TfidfVectorizer()),('bagging', BaggingClassifier(n_estimators=10)),])
model = text_clf.fit(X_train, y_train)
y_pred = model.predict(X_test)
classification_metrics(model, y_test, y_pred)


