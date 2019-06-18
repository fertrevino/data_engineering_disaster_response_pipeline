import sys
import re
from sqlalchemy import create_engine
from nltk.corpus import stopwords
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import classification_report
import pickle

def load_data(database_filepath):
    """Reads the disaster response messsages stored in the database"""    
    table_name = 'messages_categories'
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(table_name, engine)
    X = df.message.values
    Y = df.iloc[:, 4:].values
    category_names = (df.columns[4:]).tolist()
    # category_names = Y.columns 
    return X, Y, category_names

def tokenize(text):
    """Tokenize the text message usint lemmatizer"""
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    # words = [w for w in tokens if w not in stopwords.words("english")]
    words = []
    for token in tokens:
        if token not in stopwords.words("english"):
            words.append(token)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for word in words:
        clean_word = lemmatizer.lemmatize(word).strip()
        clean_tokens.append(clean_word)
    return clean_tokens

def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(), n_jobs=1)),
    ])
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'tfidf__use_idf': [True, False],
        'tfidf__norm': ['l1', 'l2']
    }
    
    return GridSearchCV(pipeline, param_grid=parameters, verbose=2)

def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluates the performance of the trained model and computes accuracy"""
    Y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        print("Label:", category_names[i])
        print(classification_report(Y_test[:, i], Y_pred[:, i]))
    # print(classification_report(Y_pred, Y_test.values, target_names=category_names))
    # print('Accuracy Score: {}'.format(np.mean(Y_test.values == Y_pred)))

def save_model(model, model_filepath):
    """Saves the trained model"""
    with open(model_filepath, 'wb') as output_model_file:
        pickle.dump(model, output_model_file)


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