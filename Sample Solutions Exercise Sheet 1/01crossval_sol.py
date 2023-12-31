#!/usr/bin/env python3

# do not use any other imports!
import numpy as np
import matplotlib.pyplot as plt
import random as rd
from sklearn.neighbors import KNeighborsClassifier as BlackBoxClassifier
from sklearn.datasets import load_iris

class Evaluation:
    """This class provides functions for evaluating classifiers """

    def generate_cv_pairs(self, n_samples, n_folds=5, n_rep=1, rand=False,
                          strat=False):
        """ Train and test pairs according to k-fold cross validation

        Parameters
        ----------

        n_samples : int
            The number of samples in the dataset

        n_folds : int, optional (default: 5)
            The number of folds for the cross validation

        n_rep : int, optional (default: 1)
            The number of repetitions for the cross validation

        rand : boolean, optional (default: False)
            If True the data is randomly assigned to the folds. The order of the
            data is maintained otherwise. Note, *n_rep* > 1 has no effect if
            *random* is False.

        y : array-like, shape (n_samples), optional (default: None)
            If not None, cross validation is performed with stratification and
            y provides the labels of the data.

        Returns
        -------

        cv_splits : list of tuples, each tuple contains two arrays with indices
            The first array corresponds to the training data, the second to the
            testing data for the current split. The list has the length of
            *n_folds* x *n_rep*.

        """
        iris_data = load_iris()  
        X = iris_data.data 
        y = iris_data.target
        
        pairs = []
        
        # indices for normal cross validation
        indices = list(range(n_samples))
        # for stratification we need to seperate the classes
        class_idcs = [np.where(y == c)[0] for c in range(len(np.unique(y)))]
        
        for _ in range(n_rep):
            if strat:
                if rand:
                    class_idcs = [np.random.permutation(c_idcs) for c_idcs in class_idcs]
                
                # split every class's indices into n_samples parts
                c_idcs_split = ([(np.array_split(c, n_folds)) for c in class_idcs])
                
                # combine the splits from every class to form n folds
                folds = []
                for (c1, c2, c3) in zip(*c_idcs_split):
                    folds.append(np.concatenate((c1, c2 ,c3)))

            else:
                if rand:
                    rd.shuffle(indices)
                    
                folds = np.array_split(indices, n_folds)
                
            # iterate through the folds and create the cross validation pairs
            for f_idx in range(n_folds):
                test_fold   = folds[f_idx]
                train_folds = np.concatenate(np.delete(folds, f_idx, axis=0))
                pairs.append((train_folds, test_fold))
            
            if not rand:
                # n_rep should have no effect if rand is false
                break

        return pairs
        
           
    def apply_cv(self, X, y, train_test_pairs, classifier):
        """ Use cross validation to evaluate predictions and return performance

        Apply the metric calculation to all test pairs

        Parameters
        ----------

        X : array-like, shape (n_samples, feature_dim)
            All data used within the cross validation

        y : array-like, shape (n-samples)
            The actual labels for the samples

        train_test_pairs : list of tuples, each tuple contains two arrays with indices
            The first array corresponds to the training data, the second to the
            testing data for the current split

        classifier : function
            Function that trains and tests a classifier and returns a
            performance measure. Arguments of the functions are the training
            data, the testing data, the correct labels for the training data,
            and the correct labels for the testing data.

        Returns
        -------

        performance : float
            The average metric value across train-test-pairs
        """
        
        accuracies = []
        
        for train, test in train_test_pairs:
            # Split the data into training and testing sets
            x_train = X[train]
            y_train = y[train]
            
            x_test = X[test]
            y_test = y[test]
            print(len(train), len(test))
            # Train and test the classifier and save the performance measure
            accuracy = classifier(x_train, x_test, y_train, y_test)
            accuracies.append(accuracy)
           
        return np.average(accuracies)


    def black_box_classifier(self, X_train, X_test, y_train, y_test):
        """ Learn a model on the training data and apply it on the testing data

        Parameters
        ----------

        X_train : array-like, shape (n_samples, feature_dim)
            The data used for training

        X_test : array-like, shape (n_samples, feature_dim)
            The data used for testing

        y_train : array-like, shape (n-samples)
            The actual labels for the training data

        y_test : array-like, shape (n-samples)
            The actual labels for the testing data

        Returns
        -------

        accuracy : float
            Accuracy of the model on the testing data
        """
        bbc = BlackBoxClassifier(n_neighbors=10)
        bbc.fit(X_train, y_train)
        acc = bbc.score(X_test, y_test)
        return acc

if __name__ == '__main__':
    # Instance of the Evaluation class
    eval = Evaluation()

    iris = load_iris()
  
    # Problem 1.1 b) i)
    train_test_pairs = eval.generate_cv_pairs(n_samples=len(iris.data), n_folds=10)
    performance = eval.apply_cv(iris.data, iris.target, train_test_pairs, eval.black_box_classifier)
    print(f'10-fold cv: {"Average accuracy":>36}: {performance:.1%}')


    # Problem 1.1 b) ii)
    train_test_pairs = eval.generate_cv_pairs(n_samples=len(iris.data), n_folds=10, n_rep=10, rand=True)
    performance = eval.apply_cv(iris.data, iris.target, train_test_pairs, eval.black_box_classifier)
    print(f'10 x 10-fold cv (rand): {"Average accuracy:":>25} {performance:.1%}')
    
    # Problem 1.1 b) iii)
    train_test_pairs = eval.generate_cv_pairs(n_samples=len(iris.data), n_folds=10, n_rep=10, rand=True, strat=True)
    performance = eval.apply_cv(iris.data, iris.target, train_test_pairs, eval.black_box_classifier)
    print(f'10 x 10-fold cv (rand + strat): {"Average accuracy:"} {performance:.1%}')
    print(len(train_test_pairs[0][0]))