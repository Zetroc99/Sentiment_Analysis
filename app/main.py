import SentimentModel
import pickle


def main():
    """
    Generate the most optimal logistic regression model found using
    GridSearchCV with various combinations of hyperparameters.
    """
    params_grid = [
        {'solver': ['lbfgs', 'liblinear'], 'max_iter': [75, 100, 125],
         'warm_start': [False, True]}
    ]
    model = SentimentModel.SentimentModel(index_col=0, param_grid=params_grid)
    # changes target values to '-1' or '+1'
    # model.binarize(col_name='airline_sentiment')
    SentimentModel.clean(model.df)
    model.split(0.3)
    model.vectorize(ngram_range=(1, 2)) # vectorize words and word couples
    model.fit()

    # pickle logistic reg model for usage in api endpoint
    pkl_filename = "../model/airline_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)
