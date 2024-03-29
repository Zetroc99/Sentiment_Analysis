import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import GridSearchCV


class SentimentModel:

    def __init__(self, data_file="./data/airline_sentiment_analysis.csv",
                 index_col=None, param_grid=None):
        """

        :param data_file: filepath takes in similar airline datasets
        :param index_col: int that allows proper index when reading file
        :param param_grid: list containing hyperparameters to use in case
            GridSearchCV is used
        """
        self.df = pd.read_csv(data_file, index_col=index_col)
        if param_grid is None:
            self.log_reg = LogisticRegression()
        else:
            gs_lr = LogisticRegression()
            self.log_reg = GridSearchCV(gs_lr, param_grid=param_grid, cv=5,
                                        scoring='accuracy')
        self.is_vectorized = False

    def binarize(self, col_name: str, neg_label=-1):
        """
        Label binarizer to transform target labels form string to int values of
        -1 and 1
        :param col_name: str value of the column we want to transform
        :param neg_label: what value should be asssigned to negative label;
            default value is -1
        :return: pd.Series() with new binarized target values
        """
        bi = LabelBinarizer(neg_label=neg_label)
        bi.fit(self.df[col_name])
        binarized = bi.transform(self.df[col_name])
        self.df[col_name] = binarized

    def split(self, test_size, random_state=42):
        """
        Implements sklearn's train_test_split to split the data
        :param test_size: int for the size of testing set
        :param random_state: int for the random seed; default set to 42
        """
        X = self.df['text']
        y = self.df['airline_sentiment']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state)

    def vectorize(self, ngram_range=(1, 1), strip_accents='ascii'):
        """
        Vectorize word corpus in order to pass through training models
        :param ngram_range: tuple of n-gram range to make on certain words
        beings used near each other; default is 1 word
        :param strip_accents: str on whether and how to strip accents
        """
        self.vectorizer = CountVectorizer(ngram_range=ngram_range,
                                          strip_accents=strip_accents)
        self.X_train = self.vectorizer.fit_transform(self.X_train)
        self.X_test = self.vectorizer.transform(self.X_test)
        self.is_vectorized = True

    def fit(self):
        """
        Fits model after input values have been vectorized
        """
        assert self.is_vectorized, "Must vectorize (use self.vectorize())"
        self.model = self.log_reg.fit(self.X_train, self.y_train)

    def predict(self):
        """
        Predicts based on the testing set features
        :return predicted target value
        """
        prediction = self.log_reg.predict(self.X_test)
        return prediction

    def predict(self, x):
        """
        Overloaded; predicts from external inputs not contained within the
            original data set.
        :param x:list of strings
        :return predicted target value
        """
        prediction = self.log_reg.predict(self.vectorizer.transform(x))
        return prediction


def clean(df):
    """
    Helper method to clean the data a bit. Removes the "@Airline" part at the
        beginning of the text reviews and creates a new column named 'Airline'
    :param df: pd.DataFrame we want to clean
    """
    df['Airline'] = df['text'].apply(lambda x: x.split(" ")[0].replace("@", ""))
    df['text'] = df['text'].apply(lambda x: " ".join(x.split(" ")[1:]))