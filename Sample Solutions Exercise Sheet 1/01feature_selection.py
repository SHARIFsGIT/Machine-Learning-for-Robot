import numpy as np
import itertools
from sklearn.datasets import load_iris

def load_dataset():
    """
    Loads the iris data set and returns its features and labels.
    """
    X, y = load_iris(return_X_y=True)

    return X, y


def calculate_abs_correlation(X, y):
    """
    Calculates the feature-feature correlation and the feature-class correlation of the provided data.
    """
    correlation_matrix = np.abs(np.corrcoef(X, y, rowvar=False))

    return correlation_matrix


def calculate_merit(X, y, feature_set):
    """
    Calculates the merit for the given feature set.
    """
    correlations = calculate_abs_correlation(X, y)
    cardinality = len(feature_set)

    if cardinality < 2:
        rff = 1.0
    else:
        rff = np.mean([correlations[pair] for pair in itertools.combinations(feature_set, 2)])

    rfc = np.mean(correlations[feature_set, -1])

    return cardinality * rfc / np.sqrt(cardinality + (cardinality-1) * cardinality * rff)


def greedy_cfs(X, y):
    """
    Performs greedy hill-climbing feature selection based on merit.
    """

    unselected_features = list(range(X.shape[1]))
    selected_features = []

    last_merit = 0

    while True:

        best_merit = 0
        best_feature = None

        print(f"current set: {selected_features}")
        print(f"options: {unselected_features}")

        for f in unselected_features:
            # calculate the merit of all next subsets
            merit = calculate_merit(X, y, selected_features + [f])

            print(f"merit for set {selected_features + [f]} is {merit}")

            # check if new best merit was found
            if merit > best_merit:
                best_merit = merit
                best_feature = f

        # check for termination
        if best_merit <= last_merit:
            return selected_features
        else:
            # housekeeping
            selected_features.append(best_feature)
            unselected_features.remove(best_feature)
            last_merit = best_merit


def main():
    """
    Run feature selection.
    """
    X, y = load_dataset()

    selected_features = greedy_cfs(X, y)

    print(f"The selected features are: {selected_features}.")

    print(calculate_merit(X,y, [0,1,2]))

if __name__ == "__main__":
    main()