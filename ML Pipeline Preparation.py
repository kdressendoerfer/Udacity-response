#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# In[1]:


# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import pickle
import nltk

nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

import warnings

warnings.simplefilter('ignore')


# In[2]:


# load data from database
engine = create_engine('sqlite:///Messages.db')
df = pd.read_sql("SELECT * FROM Messages", engine)

X = df['message']
Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)


# ### 2. Write a tokenization function to process your text data

# In[3]:


def tokenize(text):
    """Normalize, tokenize and stem text string
    
    Args:
    text: string. String containing message for processing
       
    Returns:
    stemmed: list of strings. List containing normalized and stemmed word tokens
    """
    # Convert text to lowercase and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize words
    tokens = word_tokenize(text)
    
    # Stem word tokens and remove stop words
    stemmer = PorterStemmer()
    stop_words = stopwords.words("english")
    
    stemmed = [stemmer.stem(word) for word in tokens if word not in stop_words]
    
    return stemmed


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

# In[4]:


pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
])


# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# In[5]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 1)

np.random.seed(17)
pipeline.fit(X_train, Y_train)


# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[6]:


def get_eval_metrics(actual, predicted, col_names):
    """Calculate evaluation metrics for ML model
    
    Args:
    actual: array. Array containing actual labels.
    predicted: array. Array containing predicted labels.
    col_names: list of strings. List containing names for each of the predicted fields.
       
    Returns:
    metrics_df: dataframe. Dataframe containing the accuracy, precision, recall 
    and f1 score for a given set of actual and predicted labels.
    """
    metrics = []
    
    # Calculate evaluation metrics for each set of labels
    for i in range(len(col_names)):
        accuracy = accuracy_score(actual[:, i], predicted[:, i])
        precision = precision_score(actual[:, i], predicted[:, i])
        recall = recall_score(actual[:, i], predicted[:, i])
        f1 = f1_score(actual[:, i], predicted[:, i])
        
        metrics.append([accuracy, precision, recall, f1])
    
    # Create dataframe containing metrics
    metrics = np.array(metrics)
    metrics_df = pd.DataFrame(data = metrics, index = col_names, columns = ['Accuracy', 'Precision', 'Recall', 'F1'])
      
    return metrics_df


# In[7]:


# Calculate evaluation metrics for training set
Y_train_pred = pipeline.predict(X_train)
col_names = list(Y.columns.values)

print(get_eval_metrics(np.array(Y_train), Y_train_pred, col_names))


# In[8]:


# Calculate evaluation metrics for test set
Y_test_pred = pipeline.predict(X_test)

eval_metrics0 = get_eval_metrics(np.array(Y_test), Y_test_pred, col_names)
print(eval_metrics0)


# The F1 is interestingly low. It seems like it is because of a unbalanced dataset:

# In[9]:


# Calculation the proportion of each column that have label == 1
Y.sum()/len(Y)


# This means we need to fit a seperate model to each of the y-columns....not ideal

# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[10]:


# Define performance metric for use in grid search scoring object
def performance_metric(y_true, y_pred):
    """Calculate median F1 score for all of the output classifiers
    
    Args:
    y_true: array. Array containing actual labels.
    y_pred: array. Array containing predicted labels.
        
    Returns:
    score: float. Median F1 score for all of the output classifiers
    """
    f1_list = []
    for i in range(np.shape(y_pred)[1]):
        f1 = f1_score(np.array(y_true)[:, i], y_pred[:, i])
        f1_list.append(f1)
        
    score = np.median(f1_list)
    return score


# In[11]:


# Create grid search object

parameters = {'vect__min_df': [1, 5],
              'tfidf__use_idf':[True, False],
              'clf__estimator__n_estimators':[10, 25], 
              'clf__estimator__min_samples_split':[2, 5, 10]}

scorer = make_scorer(performance_metric)
cv = GridSearchCV(pipeline, param_grid = parameters, scoring = scorer, verbose = 10)

# Find best parameters
np.random.seed(81)
tuned_model = cv.fit(X_train, Y_train)


# In[12]:


# Get results of grid search
tuned_model.cv_results_


# In[13]:


# Best test score
np.max(tuned_model.cv_results_['mean_test_score'])


# In[14]:


# Parameters for best test score
tuned_model.best_params_


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# In[15]:


# Calculate evaluation metrics for test set
tuned_pred_test = tuned_model.predict(X_test)

eval_metrics1 = get_eval_metrics(np.array(Y_test), tuned_pred_test, col_names)

print(eval_metrics1)


# In[16]:


# Get summary stats for first model
eval_metrics0.describe()


# In[17]:


# Get summary stats for tuned model
eval_metrics1.describe()


# As seen, everything which has been done increased the median ans mean F1 score. But yet we still have some output classifiers (50%) with a score of less than 0.24, which is because we have low recall values

# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# In[18]:


# Lets try SVM
pipeline2 = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(SVC()))
])

parameters2 = {'vect__min_df': [5],
              'tfidf__use_idf':[True],
              'clf__estimator__kernel': ['poly'], 
              'clf__estimator__degree': [1, 2, 3],
              'clf__estimator__C':[1, 10, 100]}

cv2 = GridSearchCV(pipeline2, param_grid = parameters2, scoring = scorer, verbose = 10)

# Find best parameters
np.random.seed(81)
tuned_model2 = cv2.fit(X_train, Y_train)


# In[19]:


# Get results of grid search
tuned_model2.cv_results_


# In[20]:


# Calculate evaluation metrics for test set
tuned_pred_test2 = tuned_model2.predict(X_test)

eval_metrics2 = get_eval_metrics(np.array(Y_test), tuned_pred_test2, col_names)

print(eval_metrics2)


# ### 9. Export your model as a pickle file

# In[21]:


# Pickle best model
pickle.dump(tuned_model, open('disaster_model.sav', 'wb'))


# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

# In[ ]:




