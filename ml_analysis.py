import os
import pickle
import json
import pandas as pd

from sklearn.metrics import roc_auc_score


if __name__ == '__main__':
    file_name = 'Исход_MACCE_КТ_RandomForestClassifier_roc_auc_0.721'

    file = 'models\\' + file_name + '_model.sav'
    loaded_model = pickle.load(open(file, 'rb'))

    file = 'models\\' + file_name + '_target.sav'
    target = pd.read_pickle(file)

    file = 'models\\' + file_name + '_predicts.sav'
    predicts = pd.read_pickle(file)



    score = roc_auc_score(target, loaded_model.predict(predicts), average=None)

    print(score)

    print(loaded_model.get_params(deep=True))
    print(loaded_model.decision_path(predicts))


# TODO: ROC curve
# TODO: IMPORTANCE

