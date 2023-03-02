import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
import logging
logging.getLogger('lightgbm').setLevel(logging.ERROR)
from sklearn.metrics import accuracy_score
import random
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from hyperopt import tpe, STATUS_OK, Trials, hp, fmin, STATUS_OK, space_eval
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

random.seed(42)
np.random.seed(42)


class Classifier():
    class ClassifierTrainer:
        """
        A class for training and evaluating a classifier using Bayesian optimization.

        Parameters
        ----------
        df_train : pandas.DataFrame
            A DataFrame containing the training data. The last column should contain
            the target labels.
        df_test : pandas.DataFrame
            A DataFrame containing the test data. The last column should contain the
            target labels.
            The target array of the test data.
        clf : classifier object
            The classifier object to be trained and evaluated.
        space : dict
            A dictionary of hyperparameter search space for Bayesian optimization.


        Attributes
    ----------
    df_train : pandas.DataFrame
        A DataFrame containing the training data.
    df_test : pandas.DataFrame
        A DataFrame containing the test data.
    clf : Classifier
        The classifier object.
    baes : function
        A function that creates an instance of the classifier for Bayesian optimization.
    params : dict
        A dictionary of hyperparameters for the classifier.
    space : dict
        A dictionary of hyperparameter search space for Bayesian optimization.
    X_train : pandas.DataFrame
        A DataFrame containing the feature data for the training set.
    y_train : pandas.DataFrame
        A DataFrame containing the target data for the training set.
    X_test : pandas.DataFrame
        A DataFrame containing the feature data for the test set.
    y_test : pandas.DataFrame
        A DataFrame containing the target data for the test set.
    y_pred : numpy.ndarray
        A numpy array containing the predicted target labels for the test set using the default classifier.
    accuracy : float
        The accuracy score of the default classifier on the test set.
    clf_baesian : Classifier
        The classifier object trained using Bayesian optimization.
    y_pred_baesian : numpy.ndarray
        A numpy array containing the predicted target labels for the test set using the classifier tuned with Bayesian optimization.
    accuracy_baesian : float
        The accuracy score of the classifier tuned with Bayesian optimization on the test set.

        Methods
        -------
        Methods
    -------
    split_data()
        Splits the training and test data into feature and target arrays.
    fit_predict_evaluate()
        Fits the classifier to the training data, makes predictions on the test data, and calculates the accuracy score.
    bayesian_tuning()
        Performs Bayesian optimization of the classifier's hyperparameters on the training data and stores the best hyperparameters.
    generate_data()
        Generates a DataFrame containing information about the training and evaluation of the classifier.
"""

    def __init__(self, df_train, df_test,model,space):
        """
        Initializes the Classifier object.

        Parameters
        ----------
        df_train : pandas.DataFrame
            A DataFrame containing the training data. The last column should contain the target labels.
        df_test : pandas.DataFrame
            A DataFrame containing the test data. The last column should contain the target labels.
        model : function
            A function that returns a classifier object when called.
        space : dict
            A dictionary of hyperparameter search space for Bayesian optimization.
        """
        self.df_train = df_train
        self.df_test = df_test
        self.clf = model()
        self.baes = model
        self.params = {}
        self.space = space

    def split_data(self):
        """Splits the training and test data into feature and target.

                Raises
                ------
                ValueError
                    If df_train or df_test is not a non-empty dataframe, or if the shapes
                    of df_train and df_test number of columns is not equal.

                """
        try:
            # Check if df_train and df_test are dataframes and are not empty
            if not isinstance(self.df_train, pd.DataFrame) or self.df_train.empty:
                raise ValueError("df_train is not a non-empty dataframe")
            if not isinstance(self.df_test, pd.DataFrame) or self.df_test.empty:
                raise ValueError("df_test is not a non-empty dataframe")

            # Check if the shapes of df_train and df_test are equal
            if self.df_train.shape[1] != self.df_test.shape[1]:
                raise ValueError("Input data shape is incorrect")

            # Split df_train into X_train and y_train
            self.X_train = self.df_train.iloc[:, :-1]
            self.y_train = self.df_train.iloc[:, -1]

            # Split df_test into X_test and y_test
            self.X_test = self.df_test.iloc[:, :-1]
            self.y_test = self.df_test.iloc[:, -1]

        except Exception as e:
            raise ValueError("Input data shape is incorrect")

    def fit_predict_evaluate(self):
        """
        Fits the classifier to the training data, makes predictions on the test data,
        and calculates the accuracy score of the predictions.

        Returns
        -------
        None
            This method only sets the `y_pred` and `accuracy` attributes of the object.
        """
        # Fit the classifier to the training data
        self.clf.fit(self.X_train, self.y_train)

        # Make predictions on the test data
        self.y_pred = self.clf.predict(self.X_test)

        # Calculate the accuracy score of the predictions
        self.accuracy = accuracy_score(self.y_test, self.y_pred)

    def bayesian_tuning(self):
        """
        Perform Bayesian hyperparameter tuning on the classifier using the given hyperparameter space.

        This method uses the hyperopt library to perform Bayesian optimization to find the optimal
        hyperparameters for the given classifier and hyperparameter space. It then trains the classifier
        using the optimal hyperparameters, and evaluates its performance on the test set.

        :return: None
        """

        # Objective function
        def objective(params):
            """
            The objective function that is optimized during the Bayesian optimization.

            This function trains the classifier using the given hyperparameters, and returns the
            negative of the cross-validated accuracy score, since hyperopt tries to minimize
            the objective function.

            :param params: A dictionary containing the hyperparameters to use for training the classifier.
            :return: A dictionary containing the loss (negative of the cross-validated accuracy score),
                     the hyperparameters used, and the optimization status.
            """

            model = self.baes(**params)
            kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
            scores = cross_val_score(model, self.X_train, self.y_train, cv=kfold, scoring='accuracy', n_jobs=-1)

            # Extract the best score
            best_score = max(scores)

            # Loss must be minimized
            loss = - best_score

            # Dictionary with information for evaluation
            return {'loss': loss, 'params': params, 'status': STATUS_OK}

        # Initialize the trials object for storing the results of each evaluation
        bayes_trials = Trials()

        # Run the Bayesian optimization to find the optimal hyperparameters
        best_params = fmin(fn=objective, space=self.space, algo=tpe.suggest, max_evals=10, trials=bayes_trials)
        self.params.update(space_eval(self.space, best_params))

        # Train the classifier using the optimal hyperparameters
        self.clf_baesian = self.baes(**self.params)
        self.clf_baesian.fit(self.X_train, self.y_train)

        # Make predictions on the test data using the tuned model
        self.y_pred_baesian = self.clf_baesian.predict(self.X_test)

        # Calculate the accuracy score
        self.accuracy_baesian = accuracy_score(self.y_test, self.y_pred_baesian)

    def generate_data(self):
        """
        Generate and store evaluation metrics for the classifier's default and tuned hyperparameters on the given dataset.

        The method creates a new pandas DataFrame with the following columns:
        'Data_type': a string representing the type of dataset used, either 'Basic' or 'Meta'.
        'Classifier': the class of the classifier used for training.
        'Accuracy Default score': the accuracy score of the classifier using default hyperparameters.
        'Accuracy Bayesian score': the accuracy score of the classifier using hyperparameters tuned with Bayesian optimization.
        'Hyperparameters': a string representation of the hyperparameters used by the classifier, either 'Default params' or the
                           dictionary of tuned hyperparameters.

        Returns:
        None.
        """
        # Create an empty pandas DataFrame with the required columns
        self.df = pd.DataFrame(columns=['Data_type', 'Classifier', 'Accuracy Default score', 'Accuracy Bayesian score', 'Hyperparameters'])

        # Set the 'Data_type' column value based on the size of the training data
        if self.df_train.shape[0] == 10:
            self.df.loc[0, 'Data_type'] = 'Meta'
        else:
            self.df.loc[0, 'Data_type'] = 'Basic'

        # Set the 'Classifier' column value to the class of the trained classifier
        self.df.loc[0, 'Classifier'] = type(self.clf)

        # Set the 'Accuracy Default score' and 'Accuracy Bayesian score' columns using the classifier's default and tuned hyperparameters
        self.df.loc[0, 'Accuracy Default score'] = self.accuracy
        self.df.loc[0, 'Accuracy Bayesian score'] = self.accuracy_baesian

        # Set the 'Hyperparameters' column based on which hyperparameters were used to achieve the highest accuracy
        if self.accuracy >= self.accuracy_baesian:
            self.df.loc[0, 'Hyperparameters'] = 'Default params'
        else:
            self.df.loc[0, 'Hyperparameters'] = str(self.params)


def main():
    """
    Main function that loads the data and runs the classifiers for both basic and meta datasets.
    """
    # Load df_train
    with open('df_train.pkl', 'rb') as f:
        df_train = pickle.load(f)

    # Load df_test
    with open('df_test.pkl', 'rb') as f:
        df_test = pickle.load(f)

    # Load df_train_meta
    with open('df_train_meta.pkl', 'rb') as f:
        df_train_meta = pickle.load(f)

    # Hyperparameter search space for LightGBM
    space_lgbm = {
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
        'num_leaves': hp.choice('num_leaves', np.arange(30, 150)),
        'feature_fraction': hp.uniform('feature_fraction', 0.5, 1.0),
        'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1.0)}

    # Hyperparameter search space for SVM
    space_svm = {
        'C': hp.uniform('C', 0, 20),
        'kernel': hp.choice('kernel', ['linear', 'sigmoid', 'poly', 'rbf']),
        'gamma': hp.uniform('gamma', 0, 20)}

    # Hyperparameter search space for KNN
    space_knn = {'n_neighbors': hp.choice('n_neighbors', np.arange(1, 10)),
                 'weights': hp.choice('weights', ['uniform', 'distance']),
                 'p': hp.choice('p', [1, 2])}

    # Classifiers
    model_lgbm = lgb.LGBMClassifier
    model_svc = SVC
    model_knn = KNeighborsClassifier

    # Dataframe to store the results
    df = pd.DataFrame(columns=['Data_type', 'Classifier', 'Accuracy Default score', 'Accuracy Bayesian score',
                               'Hyperparameters'])

    # Run classifiers for both basic and meta datasets
    for data_type in ['basic', 'meta']:
        if data_type == 'basic':
            # Classifier for basic dataset
            lgbm_classifier = Classifier(df_train, df_test, model_lgbm, space_lgbm)
            svm_classifier = Classifier(df_train, df_test, model_svc, space_svm)
            knn_classifier = Classifier(df_train, df_test, model_knn, space_knn)
        else:
            # Classifier for meta dataset
            lgbm_classifier = Classifier(df_train_meta, df_test, model_lgbm, space_lgbm)
            svm_classifier = Classifier(df_train_meta, df_test, model_svc, space_svm)
            knn_classifier = Classifier(df_train_meta, df_test, model_knn, space_knn)

        # Split the data into training and test sets
        lgbm_classifier.split_data()
        svm_classifier.split_data()
        knn_classifier.split_data()

        # Fit, predict and evaluate using default hyperparameters
        lgbm_classifier.fit_predict_evaluate()
        svm_classifier.fit_predict_evaluate()
        knn_classifier.fit_predict_evaluate()

        # Perform Bayesian tuning of hyperparameters
        lgbm_classifier.bayesian_tuning()
        svm_classifier.bayesian_tuning()
        knn_classifier.bayesian_tuning()

        # Generate a row of the dataframe with the results
        lgbm_classifier.generate_data()
        svm_classifier.generate_data()
        knn_classifier.generate_data()

        # Concatenate the results from all classifiers into a single dataframe

        df = pd.concat([df, lgbm_classifier.df, svm_classifier.df, knn_classifier.df])
    # Save the dataframe to csv
    df.to_csv('Comparisson.csv')

if __name__ == '__main__':
    main()


