import sys

# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import pickle

# download necessary NLTK data
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# import statements
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    
    '''
    load_data
    Load data from DB for further model training
    
    Input:
    database_filepath filepath to clean DB
    
    Returns:
    X messages
    y categorization
    category_names names of categories
    '''
    
    #define df based on it's location
    filepath = database_filepath
    engine = create_engine('sqlite:///' + filepath)
  
    df = pd.read_sql_table("CleanedDataTable", engine)
    
    #define X and y for the further model training
    
    X = df.message.values
    y = np.asarray(df[df.columns[4:]])
    category_names = df.columns[4:]
    print(X, y)
    return X, y, category_names

def tokenize(text):
    
    '''
    tokenize
    Return tokenized text from initial messages
    
    Input:
    text messages array
    
    Returns:
    tokens tokenized text
    '''
    
    # normalize case and remove punctuation
    new_text = [word.lower() for word in text]
    text = re.sub(r"[^a-zA-Z0-9]", " ", str(text))
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]
    return tokens

def build_model():
    
    '''
    build_model
    setup model for training
    
    Returns:
    pipeline
    '''
    
    #define classifier
    knn = KNeighborsClassifier(n_neighbors=3)
    
    #setup pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(knn, n_jobs=-1)),
    ])
    
    parameters = {
            'vect__max_df': (0.5, 1.0)
    }
    

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv

''' 
LIST OF PARAMETERS THAT CAN BE ADDED TO build_model () WITH HIGHER GPU:
    parameters = {
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
        'tfidf__smooth_idf': (True, False),
        'tfidf__sublinear_tf': (True, False),
        'tfidf__use_idf': (True, False),
        'clf__estimator__leaf_size': (10, 30, 50),
        'clf__estimator__n_jobs': (1, 2, 3), 
        'clf__estimator__n_neighbors': (1, 3, 50),
        'clf__estimator__p': (2, 3, 4),
    }
'''

def evaluate_model(model, X_test, Y_test, category_names):
    
    '''
    evaluate_mode
    evaluate trained model
    
    Input:
    model trained model
    X_test test messages
    Y_test test categorization
    category_names categories to test by
    
    Returns:
    classification_report
    '''
    
    # calculate Y_pred based on our model
    Y_pred = model.predict(X_test)
    
    #prepare classification_report for each category
    for i in range(len(category_names)):
        print("Category:", category_names[i],"\n", classification_report(Y_test[:, i], Y_pred[:, i]))

def save_model(model, model_filepath):
    
    '''
    save_model
    save our trained model as a pickle file
    
    Input:
    model our trained model
    database_filepath place to store model as a pickle file
    
    Returns:
    pickle file as a saved model
    '''
    
    pickle.dump(model,open(model_filepath,'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
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
