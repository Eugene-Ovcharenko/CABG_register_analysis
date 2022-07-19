from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier


# Decision Tree Classifier
def decision_tree():
    classifier = DecisionTreeClassifier()
    param_distributions = {'criterion': ["gini", "entropy", "log_loss"],
                           'splitter': ["best"],
                           'max_depth': [None, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50],
                           'min_samples_split': [2, 3, 4, 5, 10, 20],
                           'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20],
                           'min_weight_fraction_leaf': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
                           'max_features': [None, "sqrt", "log2"],
                           'random_state': [0],
                           'max_leaf_nodes': [None, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50],
                           'min_impurity_decrease': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                           'class_weight': [None, "balanced"],
                           'ccp_alpha': [0, 0.1, 0.2, 0.3, 0.4, 0.5]
                           }
    return classifier, param_distributions


# Random Forest Classifier
def random_forest():
    classifier = RandomForestClassifier()
    param_distributions = {'n_estimators': [2, 3, 4, 5, 10, 20, 30, 50, 100],
                           'criterion': ["gini", "entropy", "log_loss"],
                           'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50],
                           'min_samples_split': [2, 3, 4, 5, 10, 20],
                           'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20],
                           'min_weight_fraction_leaf': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
                           'max_features': [None, "sqrt", "log2"],
                           'random_state': [0],
                           'max_leaf_nodes': [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50],
                           'min_impurity_decrease': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                           'bootstrap': [True],
                           'class_weight': [None, "balanced"],
                           'ccp_alpha': [0, 0.1, 0.2, 0.3, 0.4, 0.5]
                           }
    return classifier, param_distributions

#  k-nearest neighbors classifier
def k_near_neighbors():
    classifier = KNeighborsClassifier()
    param_distributions = {'n_neighbors': [1, 3, 5, 10, 20],
                           'weights': ["uniform", "distance"],
                           'algorithm': ["auto", "ball_tree", "kd_tree", "brute"],
                           'leaf_size': [10, 30, 100, 200],
                           'p': [1, 2],
                           }
    return classifier, param_distributions

# Perceptron classifier
def perceptron():
    classifier = Perceptron()
    param_distributions = {'penalty': [None, "l2", "l1", "elasticnet"],
                           'alpha': [0.1, 0.5, 0.7, 1, 3, 5, 7, 10],
                           'fit_intercept': [True, False],
                           'max_iter': [10000],
                           'tol': [1e-3, 5e-3],
                           'shuffle': [True, False],
                           'random_state': [0],
                           'early_stopping': [True, False],
                           'validation_fraction':[0.2, 0.5],
                           'n_iter_no_change': [5],
                           'class_weight': [None, "balanced"],
                           'n_jobs': [-1]
                           }
    return classifier, param_distributions

# Multi-layer Perceptron classifier
def multi_layer_perceptron():
    classifier = MLPClassifier()
    param_distributions = {'hidden_layer_sizes': [(39, 20, 10, 5), (52, 35, 23, 15, 10), (29, 11, 4), (118, 78, 39, 20)],
                           'activation': ["logistic", "relu"], # ‘identity’, ‘logistic’, ‘tanh’, ‘relu’
                           'solver': ["adam"], # "lbfgs", "sgd"
                           'alpha': [0.001, 0.1, 0.5, 1, 3],
                           'batch_size': ["auto"],
                           'learning_rate_init': [0.1, 0.01, 0.001, 0.0001],
                           'max_iter': [10000],
                           'shuffle': [True],
                           'random_state': [0],
                           'tol': [1e-4, 1e-3],
                           'early_stopping': [False],
                           'validation_fraction': [0.2, 0.5],
                           'n_iter_no_change': [10],
                           'verbose': [0]
                           }
    return classifier, param_distributions