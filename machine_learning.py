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

# def ML_MultiOutputClassifier(X_train, y_train, X_test, y_test):
#     clf = MultiOutputClassifier(LogisticRegression(max_iter=1000)).fit(X_train, y_train)
#     predictions = clf.predict(X_test)
#
#     return predictions


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
    # # predictions = ML_MultiOutputClassifier(X_train, y_train, X_test, y_test)
    # clf = MultiOutputClassifier(LogisticRegression(max_iter=1000)).fit(X_train, y_train)
    # predictions = clf.predict(X_test)

    X = np.array(df_predicts)                                                           # X&y DATA define
    y = np.array(df_target.iloc[:, -1])

    cv = StratifiedKFold(n_splits=10)                                                   # Number of Folds

    # classifier = svm.SVC(kernel="linear", probability=True, random_state=42)          # classifier
    classifier = LogisticRegression(max_iter=10000)                                     # classifier

    # ML to ROC
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 1000)                                                  # ROC cirve resolution
    fig, ax = plt.subplots()
    for i, (train, test) in enumerate(cv.split(X, y)):
        classifier.fit(X[train], y[train])
        viz = RocCurveDisplay.from_estimator(classifier, X[test], y[test],
                                             name="ROC fold {}".format(i),
                                             alpha=0.3, lw=1, ax=ax)
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

    plt.show()
