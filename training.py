from os.path import exists
from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xlsxwriter
import scipy
import json
import os
import pickle
from sklearn.model_selection import RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.linear_model import LogisticRegression
from sklearn import svm

from models import decision_tree_cl
from models import random_forest_cl
from models import skl_knn_cl
from models import skl_perceptron_cl
from models import skl_mlp_cl
from models import skl_ada_boost_cl
from models import skl_bagging_cl
from models import skl_gb_cl
from models import xg_boost
from models import catboost_cl
from models import lightgbm_cl


# Random Search CV function --------------------------------------------------------------------------------------------
def random_search_cv(classifier, param_distributions, cv=5, scoring='roc_auc'):
    start = time()
    search = RandomizedSearchCV(classifier,
                                param_distributions,
                                cv=cv,
                                n_iter = 20,
                                scoring=scoring,
                                return_train_score=False,
                                random_state=None,
                                n_jobs=-1,
                                verbose=0
                                ).fit(X, y)

    score = search.best_score_
    print('Model: \33[32m{}\033[0m'.format(classifier.__class__.__name__))
    print(search.scoring, 'SCORE: \033[91m{0:.3f}\033[00m'.format(score))
    _time = '{0:.2f} s.'.format(time() - start)
    print('Time:', _time)
    best_params = search.best_params_
    # print(best_params)
    classifier.set_params(**best_params)
    metrics = {search.scoring: score}
    _cv = 'cv=' + str(search.cv)
    return _cv, classifier, metrics, _time


# Halving search CV function -------------------------------------------------------------------------------------------
def halving_search_cv(classifier, param_distributions, cv=5, scoring='roc_auc'):
    start = time()
    search = HalvingRandomSearchCV(classifier,
                                   param_distributions,
                                   cv=cv,
                                   factor=1.5,
                                   n_candidates=10,
                                   scoring=scoring,
                                   return_train_score=False,
                                   aggressive_elimination=False,
                                   random_state=False,
                                   n_jobs=-1,
                                   verbose=1
                                   ).fit(X, y)

    score = search.best_score_
    print('Model: \33[32m{}\033[0m'.format(classifier.__class__.__name__))
    print(search.scoring, 'SCORE: \033[91m{0:.3f}\033[00m'.format(score))
    print('Time: {0:.2f} seconds'.format(time() - start))
    best_params = search.best_params_
    classifier.set_params(**best_params)
    metrics = {search.scoring: score}
    print(best_params)
    _cv ='cv=' + str(search.cv)
    halving_search_vis(search)
    return _cv, classifier, metrics


# Plot of halving search scores of candidates over iterations function -------------------------------------------------
def halving_search_vis(search):
    results = pd.DataFrame(search.cv_results_)
    results["params_str"] = results.params.apply(str)
    results.drop_duplicates(subset=("params_str", "iter"), inplace=True)
    mean_scores = results.pivot(
        index="iter", columns="params_str", values="mean_test_score"
    )
    ax = mean_scores.plot(legend=False, alpha=0.6)

    labels = [
        f"iter={i}\nn_samples={search.n_resources_[i]}\nn_candidates={search.n_candidates_[i]}"
        for i in range(search.n_iterations_)
    ]
    ax.set_xticks(range(search.n_iterations_))
    ax.set_xticklabels(labels, rotation=45, multialignment="left")
    ax.set_title("Scores of candidates over iterations")
    ax.set_ylabel("mean test score", fontsize=15)
    ax.set_xlabel("iterations", fontsize=15)
    plt.tight_layout()
    plt.show()
    return 0


# Save regression coefficients to JSON ---------------------------------------------------------------------------------
def reg_coef_export(model, metric):
    model_param = {}
    model_param['coef'] = (model.coef_).tolist()
    model_param['intercept'] = (model.intercept_).tolist()
    file_name = 'results\\' + str(model.__class__.__name__) + '_AUC_' + str(metric) + '.json'
    json_txt = json.dumps(model_param, indent=4)
    with open(file_name, 'w') as file:
        file.write(json_txt)
    return 0


# Export ML models & dump data to a pickle file ------------------------------------------------------------------------
def export_model(model, metrics, df_target, df_predicts):
    path = 'models'                                                                     # check(create) "models" folder
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

    target_name = df_target.name.replace(" ", "_")
    metrics =[(key + '_'+str(round(metrics[key], 3))) for key in metrics]
    filebase = 'models\\' + str(target_name) + '_' + str(model.__class__.__name__) + '_' + str(metrics[0])
    file = filebase + '_model' + '.sav'
    pickle.dump(model, open(file, 'wb'))                                                # dump model

    file = filebase + '_predicts' + '.sav'
    df_predicts.to_pickle(file)                                                         # dump predicts ~ X

    file = filebase + '_target' + '.sav'
    df_target.to_pickle(file)                                                           # dump target ~ y

    return 0


# Export models configuration and metrics to excel ---------------------------------------------------------------------
def export_results(target_name, X, cv, model, metrics, _time):
    path = 'results'                                                                    # check(create) "result" folder
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

    export_groups = {}                                                                  # Save results in dictionary
    export_groups['TARGET'] = target_name
    export_groups['DATA'] = X.shape
    export_groups['Model'] = model.__class__.__name__
    export_groups['Time'] = _time
    for key in metrics:
        export_groups[key] = round(metrics[key], 3)                                     # round metrics
    export_groups['CV'] = cv
    exprt = pd.concat([pd.Series(export_groups),                                        # concat model params
                       pd.Series(model.get_params(deep=True))],
                      axis=0)
    exprt = exprt.replace(np.nan, '_None_')                                             # replace certain params states
    exprt = exprt.replace(0, '0')
    exprt = exprt.replace(1, '1')
    exprt = exprt.replace(False, '_False_')
    exprt = exprt.replace(True, '_True_')

    file = 'results\\train_' + str(model.__class__.__name__) + '.xlsx'                  # Results file name & path

    if exists(file) == False:                                                           # check(create) "result" file
        with pd.ExcelWriter(file, engine='xlsxwriter') as writer:
            exprt.to_excel(writer, sheet_name='Results', header=False, startcol=0, startrow=0)

    imprt = pd.read_excel(file, index_col=0, header=None)                               # import stored results

    if set(exprt.index).issubset(imprt.index):                                          # check for new rows in results
        write_restriction = []                                                          # flags for rewrite permission
        for col in imprt.columns:
            compare = []                                                                # compare each row for novelty:
            for idx in exprt.index:
                if isinstance(exprt.loc[idx], (int, float, complex)) \
                        and not isinstance(exprt.loc[idx], bool):                       # - compare values
                    compare.append(exprt.loc[idx] == imprt[col].loc[idx])
                else:
                    compare.append(str(exprt.loc[idx]) == str(imprt[col].loc[idx]))     # - compare strings
            write_restriction.append(all(compare))                                      # set flags for write permission
    else:
        write_restriction = [False]                                                     # flag is F if newlines

    if any(write_restriction) is False:                                                 # rewrite if all flags are F
        exprt = pd.concat([imprt, exprt], axis=1, join="outer")                         # concat existing & new results
        with pd.ExcelWriter(file, engine='xlsxwriter') as writer:
            exprt.to_excel(writer, sheet_name='Results', header=False, startcol=0, startrow=0, na_rep='')
            writer.sheets['Results'].set_column(0, 0, 30)
            writer.sheets['Results'].set_column(1, 1000, 20)

    return 0


# ROC curves & AUC assessment function based for multiple CV folds of data classification-------------------------------
def roc_auc(X, y, cv, classifier):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 1000)                                                  # ROC cirve resolution
    fig, ax = plt.subplots(figsize=(16, 8))
    for i, (train, test) in enumerate(cv.split(X, y)):
        classifier.fit(X[train], y[train])
        viz = RocCurveDisplay.from_estimator(classifier, X[test], y[test],
                                             name="ROC fold {}".format(i),
                                             alpha=0.0, lw=1, ax=ax)                    # alpha=0.3 for ON
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], label="Chance",                                             # chance line
            linestyle="--", lw=1, color="black",  alpha=0.8
            )
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color="b", lw=2, alpha=0.8,                             # mean ROC
            label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc)
            )
    se_tpr = np.std(tprs, axis=0) / np.sqrt(i+1)                                        # confidence interval for ROC
    tprs_upper = np.minimum(mean_tpr + 2 * se_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - 2 * se_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper,
                    color="grey", alpha=0.2, label=r"0.95 CI"
                    )
    ax.set(xlim=[0, 1.05], ylim=[0, 1.05],                                              # plot prop
           title="Receiver operating characteristic curve")
    ax.legend(loc="lower right")
    plt.show()

    # TODO: tune legend
    # TODO: save fig wit name


    return mean_auc


# Main function ========================================================================================================
if __name__ == '__main__':

    # load prepared data
    excel_file = 'dataset\prepared_data.xlsx'
    df = pd.read_excel(excel_file, header=0, index_col=0)

    # separating data on DataFrames and set types
    binary_cols = df.loc[:, df.isin([0, 1]).all()]                                      # select the binary cols only
    df[binary_cols.columns] = binary_cols.astype(np.uint8, copy=False)                  # predictors binary as uint8
    non_binary_cols = df.drop(columns=binary_cols.columns, axis=1, inplace=False)       # select non binary cols only

    df_targets = df.loc[:, df.columns.str.contains('^Исход.*') == True]                 # target DataFrame cols
    df_predicts = df.drop(df_targets.columns, axis=1)                                   # all predictors DataFrame

    # Machine learning
    df_target = df_targets.iloc[:, -2]                                                 # define target
    target_name = df_target.name
    print('Target:', target_name)
    X = np.array(df_predicts)
    y = np.array(df_target)

    # Random Search CV -------------------------------------------------------------------------------------------------
    models = (random_forest_cl(),
              decision_tree_cl(),
              skl_perceptron_cl(),
              skl_mlp_cl(),
              #scikit_gb_cl(),
              skl_bagging_cl(),
              xg_boost(),
              lightgbm_cl(),
              catboost_cl(),
              )                                                                         # <- Models for optimization

    rnd_iterations = 1000                                                               # Number of cycles of rnd search
    for model in models:
        classifier, param_distributions = model                                         # reading model's parameters
        bst=[]                                                                          # list of scores
        for i in range(rnd_iterations):                                                 # cycles of random search
            print('Cycle #', i)
            _cv, classifier, metrics, _time = random_search_cv(classifier,
                                                        param_distributions,
                                                        cv=5, scoring='roc_auc')        # train and test model
            classifier.fit(X,y)

            bst.append(metrics['roc_auc'])
            file = 'results\\train_' + str(classifier.__class__.__name__) + '.xlsx'     # load previous results
            if exists(file) == False:
                export_results(target_name, X, _cv, classifier, metrics, _time)         # export results first time
            else:
                previous_results = pd.read_excel(file, index_col=0, header=None)        # load previous results

                if metrics['roc_auc'] > previous_results.loc[:,
                                        previous_results.loc['TARGET']
                                        == target_name].loc['roc_auc'].max():           # if new better then previous...
                    export_results(target_name, X, _cv, classifier, metrics, _time)     # ...export results and...
                    export_model(classifier, metrics, df_target, df_predicts)           # ...store the model
        print('\33[32m BEST SCORE of {}\033[0m: {:.3f}\n'.format(str(classifier.__class__.__name__), max(bst)))






    # TODO: Export ML models
    # TODO: Variable - importance

    # TODO: 2 Hyperopt
    # TODO: 2 Optuna
    # TODO: 3 AutoML
    # TODO: 4 PSM
    # TODO: 5 turn off dro data outside 0.10 - 0.90








    # Machine learning: CV ROC assessment ------------------------------------------------------------------------------
    cv = StratifiedKFold(n_splits=5, random_state=None)                                 # Number of Folds
    classifiers = {#LogisticRegression(max_iter=10000, solver='newton-cg'),
                   # RandomForestClassifier(bootstrap=True,
                   #                        random_state=0,
                   #                        ccp_alpha=0,
                   #                        class_weight=None,
                   #                        criterion='entropy',
                   #                        max_depth=40,
                   #                        max_features=None,
                   #                        max_leaf_nodes=50,
                   #                        max_samples=None,
                   #                        min_impurity_decrease=0,
                   #                        min_samples_leaf=8,
                   #                        min_samples_split=20,
                   #                        min_weight_fraction_leaf=0.1,
                   #                        n_estimators=5
                   #                        )
                   }
    for classifier in classifiers:
        print('Classifier model \33[32m{}\033[0m with the following parameters:\n{}'
              .format(classifier, classifier.get_params()))
        metric_auc = roc_auc(X, y, cv, classifier)                                                                          # TODO: must check
        print('AUC: \033[91m{0:.3f}\033[00m'.format(metric_auc))

        metrics = {'AUC_ROC': metric_auc}
        export_results (target_name, X, cv, classifier, metrics)
        # if classifier.__class__.__name__ == 'LogisticRegression':
        #     reg_coef_export(classifier, metric_auc)

