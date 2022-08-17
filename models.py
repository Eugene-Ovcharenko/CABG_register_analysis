from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import xgboost as xgb


# Decision Tree Classifier
def decision_tree_cl():
    classifier = DecisionTreeClassifier()
    param_distributions = {
        'criterion': ["gini", "entropy", "log_loss"],
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
def random_forest_cl():
    classifier = RandomForestClassifier()
    param_distributions = {
        'n_estimators': [2, 3, 4, 5, 10, 20, 30, 50, 100],
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


# Scikit Learn bagging classifier
def skl_bagging_cl():
    classifier = BaggingClassifier()
    param_distributions = {
        'base_estimator': [None],
        'n_estimators': [5, 10, 20],
        'max_samples': [0.2, 0.5, 0.8, 1.0],
        'max_features': [0.2, 0.5, 0.8, 1.0],
        'bootstrap': [True, False],
        'random_state': [0],
        'n_jobs': [-1]
    }
    return classifier, param_distributions


#  k-nearest neighbors classifier
def skl_knn_cl():
    classifier = KNeighborsClassifier()
    param_distributions = {
        'n_neighbors': [1, 3, 5, 10, 20],
        'weights': ["uniform", "distance"],
        'algorithm': ["auto", "ball_tree", "kd_tree", "brute"],
        'leaf_size': [10, 30, 100, 200],
        'p': [1, 2]
    }
    return classifier, param_distributions


# Perceptron classifier
def skl_perceptron_cl():
    classifier = Perceptron()
    param_distributions = {
        'penalty': [None, "l2", "l1", "elasticnet"],
        'alpha': [0.1, 0.5, 0.7, 1, 3, 5, 7, 10],
        'fit_intercept': [True, False],
        'max_iter': [10000],
        'tol': [1e-3, 5e-3],
        'shuffle': [True, False],
        'random_state': [0],
        'early_stopping': [True, False],
        'validation_fraction': [0.2, 0.5],
        'n_iter_no_change': [5],
        'class_weight': [None, "balanced"],
        'n_jobs': [-1]
    }
    return classifier, param_distributions


# Multi-layer Perceptron classifier
def skl_mlp_cl():
    classifier = MLPClassifier()
    param_distributions = {
        'hidden_layer_sizes': [(39, 20, 10, 5), (52, 35, 23, 15, 10), (29, 11, 4), (118, 78, 39, 20)],
        'activation': ["logistic", "relu"],  # ‘identity’, ‘logistic’, ‘tanh’, ‘relu’
        'solver': ["adam"],  # "lbfgs", "sgd"
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


# Scikit Learn AdaBoost classifier
def skl_ada_boost_cl():
    classifier = AdaBoostClassifier()
    param_distributions = {
        'base_estimator': [None],
        'n_estimators': [10],
        'learning_rate': [0.01],
        'algorithm': ['SAMME.R'],
        'random_state': [0]
    }
    return classifier, param_distributions


# Scikit Learn Gradient Boosting Classifier
def skl_gb_cl():
    classifier = GradientBoostingClassifier()
    param_distributions = {'loss': ["log_loss", "exponential"],
                           'learning_rate': [0.1, 0.01, 0.001],
                           'n_estimators': [5, 10, 20],
                           'min_samples_split': [2, 3, 4, 5, 10, 20],
                           'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20],
                           'min_weight_fraction_leaf': [0, 0.1, 0.2],
                           'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50],
                           'random_state': [0],
                           'max_features': [None],
                           'max_leaf_nodes': [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50],
                           'n_iter_no_change': [10],
                           'tol': [1e-3, 1e-4],
                           'ccp_alpha': [0],
                           'verbose': [0],
                           'random_state': [0]
                           }
    return classifier, param_distributions


# XGBoost Classifier
def xg_boost():
    classifier = xgb.XGBClassifier()
    param_distributions = {
        'booster': ["gbtree", "dart"],
        'eta': [0.1, 0.01, 0.001],
        'gamma': [0.1, 0.01, 0.001],
        'max_depth': [2, 3, 4, 5, 10, 20, 50],
        'min_child_weight': [2, 3, 4, 5, 10, 20, 50],
        'max_delta_step': [2, 3, 4, 5, 10, 20, 50],
        'subsample': [0.5, 1],
        'sampling_method': ["uniform"],
        'colsample_bytree': [1],
        'colsample_bylevel': [1],
        'colsample_bynode': [1],
        'lambda': [0, 1],
        'alpha': [0, 1],
        'tree_method': ["auto", "exact"],
        'sketch_eps': [0.03],
        'scale_pos_weight': [1],
        'process_type': ["default"],
        'sample_type': ["uniform", "weighted"],
        'normalize_type': ["tree", "forest"],
        'rate_drop': [0, 0.3, 0.5],
        'one_drop': [0, 1],
        'skip_drop': [0, 0.33, 0.5],
        # 'updater': [],
        # 'refresh_leaf': [1],
        # 'grow_policy': [],
        # 'max_leaves': [],
        # 'max_bin': [],
        # 'predictor': [],
        # 'num_parallel_tree': [],
        # 'monotone_constraints': [],
        # 'interaction_constraints': [],
        'verbosity': [0],
        'random_state': [0]
    }
    return classifier, param_distributions


def catboost_cl():
    classifier = CatBoostClassifier()
    param_distributions = {
        'depth': [4, 5, 6, 7, 8, 9, 10],
        'learning_rate': [0.01, 0.02, 0.03, 0.04],
        'iterations': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'loss_function': ["Logloss"],
        'scale_pos_weight': [0.01, 0.1, 0.5, 1.0],
        'verbose': [False],
        'random_state': [0]
    }
    return classifier, param_distributions


def lightgbm_cl():
    classifier = LGBMClassifier()
    param_distributions = {
        'num_leaves': [30, 50, 100, 150],
        'min_data_in_leaf': [30, 50, 100, 150],
        'bagging_fraction': [0.7, 0.8, 0.9],
        'feature_fraction': [0.7, 0.8, 0.9],
        'lambda_l1': [0, 1, 10],
        'verbose_eval': [False],
        'verbose': [-1],
        'random_state': [0]
    }
    return classifier, param_distributions
