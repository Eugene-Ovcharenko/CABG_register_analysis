import os
import re
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import DetCurveDisplay, RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix


def import_excel_files(path, regex):

    files_list = [file for file in os.listdir(path) if
                  os.path.isfile(os.path.join(path, file))]
    r = re.compile(regex)                                                               # by regex ^train_*
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

    return results



if __name__ == '__main__':
    # style for plots
    plt.style.use('seaborn')
    sns.set_context("talk") # \ "paper" \ "poster" \ "notebook"
    sns.set_style("whitegrid")
    cmap = plt.cm.get_cmap('cool')

    # import
    path = 'results\\'                                                                  # files path
    regex = '^train_*'                                                                  # files' names
    results = import_excel_files(path, regex)

    all_scores = pd.DataFrame(index=['TARGET', 'Model'])                                # empty score file create

    for target in results.loc['TARGET'].unique():                                       # targets loop
        fig, ax = plt.subplots(figsize=(12, 6))                                         # ROC curve for each target
        colors_num = results.loc['Model'].unique().shape[0]                             # color for each line from cmap

        for n, model in enumerate(results.loc['Model'].unique()):                       # models loop
            metric = results.loc['roc_auc', (results.loc['Model'] == model) &
                                 (results.loc['TARGET'] == target)].values
            if metric.size == 0:                                                        # if no data
                print('\033[91m Warning: No data for ', target, '|', model, '\033[0m')
            else:                                                                       # load model and datasets
                file_name = target.replace(" ", "_") + '_' + \
                            model + '_roc_auc_' + str(metric[0])
                file = 'models\\' + file_name + '_model.sav'
                clf = pickle.load(open(file, 'rb'))
                file = 'models\\' + file_name + '_target.sav'
                y = pd.read_pickle(file)
                file = 'models\\' + file_name + '_predicts.sav'
                X = pd.read_pickle(file)

                scores = cross_validate(clf, X, y, cv=5, scoring=[                      # model evaluation on CV5
                    'roc_auc',
                    'accuracy',
                    'balanced_accuracy',
                    'average_precision',
                    'precision',
                    'recall',
                    'f1',
                    'f1_micro',
                    'f1_macro',
                    'neg_log_loss'])
                scores = pd.Series(scores).apply(lambda x: round(np.mean(x), 3))        # save average score
                scores['TARGET'] = target
                scores['Model'] = model
                print(file_name, '\tTest_score\t', scores['test_roc_auc'])

                cv = StratifiedKFold(n_splits=5)                                        # CV5 for ROC plotting
                X = np.array(X)
                y = np.array(y)
                mean_fpr = np.linspace(0, 1, 1000)
                tprs = []
                aucs = []
                tn = [None] * 5                                                         # * 5 - CV
                fp = [None] * 5
                fn = [None] * 5
                tp = [None] * 5
                y_pos = [None] * 5
                y_neg = [None] * 5

                for i, (train, test) in enumerate(cv.split(X, y)):
                    clf.fit(X[train], y[train])
                    tn[i], fp[i], fn[i], tp[i] = confusion_matrix(clf.predict(X[test]), y[test]).ravel()
                    y_pos[i] = np.sum(y[test] == 1)
                    y_neg[i] = np.count_nonzero(y[test] == 0)

                    viz = RocCurveDisplay.from_estimator(clf, X[test], y[test],
                                                         name='', alpha=0, lw=1,
                                                         ax=ax, label='_nolabel_')

                    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
                    interp_tpr[0] = 0.0
                    tprs.append(interp_tpr)
                    aucs.append(viz.roc_auc)
                mean_tpr = np.mean(tprs, axis=0)
                mean_tpr[-1] = 1.0
                mean_auc = auc(mean_fpr, mean_tpr)
                print('roc_auc from ROC curve', mean_auc)
                line = ax.plot(mean_fpr, mean_tpr, lw=2,
                        label ='{string} Mean AUC = {auc:.2f} $\pm$ {std:.2f}'.
                        format(string=model, auc=mean_auc, std=np.std(aucs)))
                line[-1].set_color(cmap(n/colors_num))                                  # replace line's color

                scores['True Positive'] = tp
                scores['Y positive number'] = y_pos
                scores['True Negative'] = tn
                scores['Y negative number'] = y_neg
                scores['False Positive'] = fp
                scores['False Negative'] = fn
                scores['roc_auc2'] = mean_auc

                all_scores = pd.concat([all_scores, scores], axis=1)

            #break
        ax.plot([0, 1], [0, 1], linestyle="--", lw=1, color="gray", label='_nolabel_', alpha=0.8)
        ax.set(xlim=[-0, 1.0], ylim=[-0, 1.0], title=target + ' | Mean ROC curve')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        file = 'results\\Model_comparison_for_' + target + '.tiff'
        fig.savefig(file, dpi=200)
        #break

    all_scores.to_excel('results\\all_scores.xlsx', header=False)






    # TODO: export best results + other metrics

    # TODO: tp fp square https://towardsdatascience.com/metrics-and-python-ii-2e49597964ff

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



    # TODO: Tree graph https://russianblogs.com/article/5797349374/


    ### "Linear SVM": make_pipeline(StandardScaler(), LinearSVC(C=0.025)),