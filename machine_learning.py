import scipy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import json
# import wandb
# wandb.init(project="CABG-register-project")


def get_divisors(n):
    divisors = []
    for m in range(1, n+1):
        if n % m == 0:
            divisors.append(m)
    return divisors

# ROC and AUC calculation function for cv data classification
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
    ax.set(xlim=[0, 1.05], ylim=[0, 1.05],                                               # plot prop
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
    y = np.array(df_target.iloc[:, -1])

    cv = StratifiedKFold(n_splits=5, random_state=None)                                 # Number of Folds

    classifiers = {LogisticRegression(max_iter=10000, solver='newton-cg'),
                   LogisticRegression(max_iter=10000, solver ='liblinear')
                   #svm.SVC(kernel="linear", probability=True, random_state=42)
                   }


    for classifier in classifiers:
        print('Classifier model \33[32m{}\033[0m with the following parameters:\n{}'
              .format(classifier, classifier.get_params()))

        metric_auc = roc_auc(X, y, cv, classifier)
        print('AUC: \033[91m{0:.3f}\033[00m'.format(metric_auc))

        model_param = {}                                                                # Save the model to JSON
        model_param['coef'] = (classifier.coef_).tolist()
        model_param['intercept'] = (classifier.intercept_).tolist()
        file_name = 'results\\' + str(classifier).split('(')[0] + '_AUC_' + str(metric_auc) + '.json'
        print(file_name)
        json_txt = json.dumps(model_param, indent=4)
        with open(file_name, 'w') as file:
            file.write(json_txt)








        # wandb.config = {
        #     "learning_rate": 0.001,
        #     "epochs": 100,
        #     "batch_size": 128
        # }
        # wandb.log({"auc": auc})


    # TODO: wandb ?



    # TODO: K-Nearest Neighbors Classification
    # TODO: Decision Trees
    # TODO: RandomForestClassifier
    # TODO: Perceptron
    # TODO: Multi-layer Perceptron ?
