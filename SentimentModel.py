import numpy as np
import pandas as pd
import pickle

from fastapi import FastAPI, Request
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import GridSearchCV


# TODO use OOP concepts to train model
# TODO: use OOP concepts to read data for training
# TODO: implement inference class using fastAPI and Swagger
# TODO: implement API server on python using fastAPI
# TODO: API must accept an english text and respond wit the predicted sentiment
# TODO: integrate Swagger documentation for the Rest API endpoint
# TODO: (BONUS) containerize the model and expose a public endpoint for your API
#       using Cortex or HuggingFace

class SentimentModel:

    def __init__(self, data_file="data/airline_sentiment_analysis.csv",
                 index_col=None, param_grid=None):
        self.df = pd.read_csv(data_file, index_col=index_col)
        if param_grid is None:
            self.log_reg = LogisticRegression()
        else:
            gs_lr = LogisticRegression()
            self.log_reg = GridSearchCV(gs_lr, param_grid=param_grid, cv=5,
                                        scoring='accuracy')
        self.is_vectorized = False

    def binarize(self, col_name: str, neg_label=-1):
        bi = LabelBinarizer(neg_label=neg_label)
        bi.fit(self.df[col_name])
        binarized = bi.transform(self.df[col_name])
        self.df[col_name] = binarized

    def split(self, test_size, random_state=42):
        X = self.df['text']
        y = self.df['airline_sentiment']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state)

    def vectorize(self, ngram_range=(1, 1), strip_accents='ascii'):
        self.vectorizer = CountVectorizer(ngram_range=ngram_range,
                                          strip_accents=strip_accents)
        self.X_train = self.vectorizer.fit_transform(self.X_train)
        self.X_test = self.vectorizer.transform(self.X_test)
        self.is_vectorized = True

    def fit(self):
        assert self.is_vectorized, "Must vectorize (use self.vectorize())"
        self.model = self.log_reg.fit(self.X_train, self.y_train)

    def predict(self):
        prediction = self.log_reg.predict(self.X_test)
        return prediction

    def predict(self, x: list):
        prediction = self.log_reg.predict(self.vectorizer.transform(x))
        return prediction


def clean(df):
    df['Airline'] = df['text'].apply(lambda x: x.split(" ")[0].replace("@", ""))
    df['text'] = df['text'].apply(lambda x: " ".join(x.split(" ")[1:]))


if __name__ == '__main__':
    params_grid = [
        {'solver': ['lbfgs', 'liblinear'], 'max_iter': [75, 100, 125],
         'warm_start': [False, True]}
    ]
    model = SentimentModel(index_col=0, param_grid=params_grid)
    model.binarize(col_name='airline_sentiment')
    clean(model.df)
    model.split(0.3)
    model.vectorize(ngram_range=(1, 2))
    model.fit()
    print(model.predict(['The flight was great']))
    print("Testing Accuracy: ", model.log_reg.score(model.X_test, model.y_test))

    pkl_filename = "model/airline_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model.log_reg, file)
