import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve, auc


def roc_curves(y_test, y_prob):

    plt.figure(figsize=(5, 5))
    plt.title('Receiver Operating Characteristic')

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))


    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.axis('tight')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    return 0


if __name__ == '__main__':
    file_name = 'Исход_MACCE_ИМ_CatBoostClassifier_roc_auc_0.568'

    file = 'models\\' + file_name + '_model.sav'
    model = pickle.load(open(file, 'rb'))

    file = 'models\\' + file_name + '_target.sav'
    target = pd.read_pickle(file)

    file = 'models\\' + file_name + '_predicts.sav'
    predicts = pd.read_pickle(file)


    # model.fit(predicts, target)

    # y_prob = model.predict(predicts)
    #
    # score = roc_auc_score(target, y_prob, average=None)
    #
    # print(score)

    print(model.get_params(deep=True))



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