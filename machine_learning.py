from os.path import exists
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xlsxwriter
import scipy
import json

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import StratifiedKFold


from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

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

# Export models configuration and metrics to excel ---------------------------------------------------------------------
def export_results(cv, model, metrics):
    export_groups = {}                                                                  # Load results in dictionary
    for key in metrics:
        export_groups[key] = round(metrics[key], 3)                                     # round metrics
    export_groups['Model'] = model.__class__.__name__
    export_groups['CV'] = cv
    exprt = pd.concat([pd.Series(export_groups),                                        # concat model params
                       pd.Series(model.get_params(deep=True))],
                      axis=0)
    exprt = exprt.replace(np.nan, '_None_')                                             # replace certain params states
    exprt = exprt.replace(False, '_False_')
    exprt = exprt.replace(True, '_True_')

    file = 'results\\train_' + str(model.__class__.__name__) + '.xlsx'                  # Results file name & path

    if exists(file) == False:                                                           # create if file doesn't exist
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
    # plt.show()

    # TODO: tune legend
    # TODO: save fig wit name
    # TODO: save AUC to table

    return mean_auc



if __name__ == '__main__':

    # load prepared data
    excel_file = 'dataset\prepared_data.xlsx'
    df = pd.read_excel(excel_file, header=0, index_col=0)

    # separating data on DataFrames
    df_target = df.loc[:, df.columns.str.contains('^Исход.*') == True]                  # targets DataFrame
    # df_target = df.loc[:, df.columns.str.contains('Вид.*') == True]

    binary_cols = df.isin([0, 1]).all()                                                 # predictors binary DataFrame
    df_binary = df[binary_cols[binary_cols].index]
    df_binary = df_binary.drop(df_target.columns, axis=1)

    float_mask = df.isin([0, 1]).all() == False                                         # predictors float DataFrame
    df_float = df[float_mask[float_mask].index].astype(np.float64)

    df_predicts = df.drop(df_target.columns, axis=1)                                    # all predictors DataFrame

    # Machine learning
    X = np.array(df_predicts)                                                           # X&y DATA define
    y = np.array(df_target.iloc[:, -2])
    # TODO: export x, y shapes and comments



    # TODO: Halving Random Search CV
    classifier = tree.DecisionTreeClassifier(random_state=0)
    param_distributions = {'criterion' : ["gini", "entropy", "log_loss"],
                           'splitter': ["best"],
                           'max_depth': [ None, 2, 3, 5, 10, 20, 30, 50],
                           'min_samples_split': [2, 3, 5, 10, 20],
                           'min_samples_leaf': [1, 2, 3, 5, 10],
                           'min_weight_fraction_leaf': [0, 0.1, 0.3, 0.5],
                           'max_features': [None, "sqrt", "log2"],
                           'random_state': [0],
                           'max_leaf_nodes': [None, 2, 5, 10, 20, 50],
                           'min_impurity_decrease': [0.0],
                           'class_weight': [None, "balanced"],
                           'ccp_alpha': [0, 0.1, 0.3]
                          }
    search = HalvingRandomSearchCV(classifier,
                                   param_distributions,
                                   cv=5,
                                   factor=1.5,
                                   n_candidates=10,
                                   scoring='roc_auc',
                                   return_train_score=False,
                                   aggressive_elimination=False,
                                   random_state=0,
                                   n_jobs=-1,
                                   verbose=1
                                   ).fit(X, y)

    score = search.best_score_
    print('Model: \33[32m{}\033[0m'.format(classifier.__class__.__name__))
    print(search.scoring, 'SCORE: \033[91m{0:.3f}\033[00m'.format(score))

    best_params = search.best_params_
    classifier.set_params(**best_params)
    metrics = {search.scoring: score}
    export_results('cv=5', classifier, metrics)
    print(best_params)

    # halving_search_vis(search)




    # Machine learning: CV ROC assessment
    # cv = StratifiedKFold(n_splits=5, random_state=None)                                 # Number of Folds
    # classifiers = {LogisticRegression(max_iter=10000, solver='newton-cg'),
    #                LogisticRegression(max_iter=10000, solver ='liblinear'),
    #                # svm.SVC(kernel="linear", probability=True, random_state=42),
    #                # tree.DecisionTreeClassifier(random_state=0),
    #                # tree.DecisionTreeClassifier(random_state=1),
    #                # RandomForestClassifier(random_state=0)
    #                }
    # for classifier in classifiers:
    #     print('Classifier model \33[32m{}\033[0m with the following parameters:\n{}'
    #           .format(classifier, classifier.get_params()))
    #     metric_auc = roc_auc(X, y, cv, classifier)
    #     print('AUC: \033[91m{0:.3f}\033[00m'.format(metric_auc))
    #
    #     metrics = {'AUC_ROC': metric_auc}
    #     export_results (cv, classifier, metrics)
    #     if classifier.__class__.__name__ == 'LogisticRegression':
    #         reg_coef_export(classifier, metric_auc)














    # TODO: wandb ?

    # TODO: K-Nearest Neighbors Classification
    # TODO: Decision Trees
    # TODO: RandomForestClassifier
    # TODO: Perceptron
    # TODO: Multi-layer Perceptron ?
