import os
import re
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import DetCurveDisplay, RocCurveDisplay

# def roc_curves(y_test, y_prob):
#
#     plt.figure(figsize=(5, 5))
#     plt.title('Receiver Operating Characteristic')
#
#     fpr, tpr, _ = roc_curve(y_test, y_prob)
#     auc = roc_auc_score(y_test, y_prob)
#     plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
#
#     plt.legend(loc='lower right')
#     plt.plot([0, 1], [0, 1], linestyle='--')
#     plt.axis('tight')
#     plt.ylabel('True Positive Rate')
#     plt.xlabel('False Positive Rate')
#     plt.show()
#     return 0


if __name__ == '__main__':

    # get the list of files
    path = 'results\\'
    files_list = [file for file in os.listdir(path) if
                  os.path.isfile(os.path.join(path, file))]
    r = re.compile('^train_*')                                                          # by regex ^train_*
    files_list = list(filter(r.match, files_list))

    # concatenate all models results
    results = pd.DataFrame()
    for file in files_list:
        file = path + file
        df = pd.read_excel(file, index_col=0, header=None)
        df.set_axis(df.loc['TARGET'], axis='columns', inplace=True)
        df = df.sort_values(by='roc_auc', axis=1, ascending=False)                      # sort for drop cols except max
        df = df.loc[:, ~df.columns.duplicated()]
        results = pd.concat([results, df], axis=1)
    results = results.T.reset_index(drop=True).T                                        # drop column names
    results.to_excel('results\\all_models_results.xlsx')                                # export data

    for target in results.loc['TARGET'].unique():
        fig, [ax_roc, ax_det] = plt.subplots(1, 2, figsize=(16, 8))                     # ROC curve
        for model in results.loc['Model'].unique():
            metric = results.loc['roc_auc', (results.loc['Model'] == model) &
                                 (results.loc['TARGET'] == target)].values
            if metric.size == 0:
                print('\033[91m Warning: No data for ', target, '|', model, '\033[0m')
            else:
                file_name = target.replace(" ", "_") + '_' + \
                            model + '_roc_auc_' + str(metric[0])
                file = 'models\\' + file_name + '_model.sav'
                clf = pickle.load(open(file, 'rb'))
                file = 'models\\' + file_name + '_target.sav'
                y = pd.read_pickle(file)
                file = 'models\\' + file_name + '_predicts.sav'
                X = pd.read_pickle(file)

                score = cross_val_score(clf, X, y, scoring="roc_auc", cv=5).mean()
                print(file_name, '\tTest_score\t', score)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

                RocCurveDisplay.from_estimator(clf, X_test, y_test, ax=ax_roc, name=model)
                DetCurveDisplay.from_estimator(clf, X_test, y_test, ax=ax_det, name=model)

            ax_roc.set_title('Receiver Operating Characteristic (ROC) curves: ' + target)
            ax_roc.grid(linestyle="--")
            ax_det.set_title('Detection Error Tradeoff (DET) curves: ' + target)
            ax_det.grid(linestyle="--")

        plt.legend()
        plt.show()

    # print(model.get_params(deep=True))



    # TODO: ROC curve
    # xx = roc_curves(target, y_prob)

    # TODO: IMPORTANCE
    # feature_scores = pd.Series(model.feature_importances_, index=predicts.columns).sort_values(ascending=False)
    # feature_scores = feature_scores[feature_scores != 0]
    # fig, ax = plt.subplots(figsize=(10, 5))
    # ax = sns.barplot(x=feature_scores, y=feature_scores.index)
    # ax.set_title("Visualize feature scores of the features")
    # ax.set_yticklabels(feature_scores.index)
    # ax.set_xlabel("Feature importance score")
    # ax.set_ylabel("Features")
    # plt.tight_layout()
    # plt.show()

    # TODO: Cycle for all models from list/ or best in table!
    # TODO:  check all models
    # TODO: Tree graph https://russianblogs.com/article/5797349374/


    ### "Linear SVM": make_pipeline(StandardScaler(), LinearSVC(C=0.025)),