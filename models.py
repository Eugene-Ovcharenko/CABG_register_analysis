from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# Decision Tree Classifier
def decision_tree():
    classifier = DecisionTreeClassifier(random_state=0)
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
    classifier = RandomForestClassifier(random_state=0)
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


def k_near_neighbors():
    classifier = KNeighborsClassifier()
    param_distributions = {'n_neighbors': [1, 3, 5, 10, 20],
                           'weights': ["uniform", "distance"],
                           'algorithm': ["auto", "ball_tree", "kd_tree", "brute"],
                           'leaf_size': [10, 30, 100, 200],
                           'p': [1, 2],
                           }
    return classifier, param_distributions
