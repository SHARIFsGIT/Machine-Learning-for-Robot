from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression as AnotherBBClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np




def load_dataset(seed=42):
    """
    Creates an artificial binary classification dataset and 
    returns its features and labels.
    """
    X, y = make_classification(n_samples=1000, n_classes=2, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=seed)

    return X_train, y_train, X_test, y_test



def roc_curve(y, probs, pos_label = 0):
    """
    Implement the receiver operation characteristics function

    @param y: array of the true labels of the samples

    @param probs: array of the model result

    @param pos_label: number of the positive label
    """
    # Sort the arguments by the other column so that the first column is in descending order
    order = probs[:,1 - pos_label].argsort()

    # Order the labels and the thresholds.
    y_ordered = y[order]
    thresholds = np.concatenate([[np.inf], probs[order, pos_label]])

    # Create output arrays
    fpr = np.zeros(len(y)+1)
    tpr = np.zeros(len(y)+1)

    # P = TP + FN 
    P = (y == pos_label).sum()
    # N = TN + FP
    N = len(y) - P


    for i, prob in enumerate(thresholds):
      # get the number of true positives and false positives
      tp = (y_ordered == pos_label)[:i].sum()
      fp = (y_ordered == 1 - pos_label)[:i].sum()

      # calculate the rates
      tpr[i] = tp / P
      fpr[i] = fp / N

    return fpr, tpr, thresholds



def calculate_auc(fpr, tpr):
    """
    Implement a function to calculate the auc value from the roc results
    and returns the best threshold.

    @param fpr: array of the false positive rates

    @param tpr: array of the true positive rates
    """

    auc = 0

    # simply calculate the area of the rectangles
    for i in range(1, len(fpr)): 
      auc += (fpr[i] - fpr[i-1]) * tpr[i]

    return auc



def another_bb_classifier(X_train, y_train, X_test):
        """ 
        Learn a model on the training data and apply it on the testing data

        @param X_train: array of training features

        @param y_train: array of training labels

        @param X_test: array of test features

        @return: array of model scores
        """
        abbc = AnotherBBClassifier(solver='lbfgs')
        abbc.fit(X_train, y_train)
        
        return abbc.predict_proba(X_test) 



def your_random_classifier(X_test):
        """
        Your implementation of a random classifier
        
        @param X_test: array of test features

        @return: array of model scores
        """

        # One way is to give every sample the same label
        random_class = np.zeros((len(X_test), 2))
        random_class[:, 0] = 1
        return random_class



def plot_roc(fpr, tpr, classifier=""):
        """
        Plot the given ROC curve

        @param fpr: array of the false positive rates

        @param tpr: array of the true positive rates
        """
        auc = calculate_auc(fpr, tpr)

        plt.plot(fpr, tpr, linestyle='--', color='darkorange', lw = 1, label='ROC curve', clip_on=False)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC curve for {classifier}, AUC = %.2f'%auc)
        plt.legend(loc="lower right")
        plt.savefig('AUC_example.png')
        plt.show()

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_dataset()

    probs = your_random_classifier(x_test)
    fpr, tpr, _ = roc_curve(y_test, probs)

    plot_roc(fpr,tpr, "Random classifier")

    probs = another_bb_classifier(x_train, y_train, x_test)
    fpr, tpr, _ = roc_curve(y_test, probs)

    plot_roc(fpr,tpr, "ABB classifier")