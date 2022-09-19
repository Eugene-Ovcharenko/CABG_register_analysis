import os
import pandas as pd
import numpy as np
import re
import joblib
import logger
import json
from supervised import AutoML
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier


pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_colwidth', None)


def get_leaderboard(target, aml_folder, aml_modes, load_mode = 'all'):
    leaderboard = pd.DataFrame()
    for mode in aml_modes:
        path = aml_folder + '\AutoML_' + target + '_' + mode
        df = pd.read_csv(path + '\leaderboard.csv')
        df.sort_values(by=['model_type', 'metric_value'], ascending=False, inplace=True)
        df['features_options'] = df['name'].apply(
            lambda n: re.findall('GoldenFeatures', n)[0] if re.findall('GoldenFeatures', n) else "")
        df.drop(df[df['features_options'] == 'GoldenFeatures'].index, inplace=True)                     # drop golden features # TODO: add Golden Features
        df.drop_duplicates(subset=['model_type'], keep='first', inplace=True)
        df.reset_index(drop=True, inplace=True)
        df['aml_mode'] = mode
        df['path'] = path + '\\' + df['name']
        leaderboard = pd.concat([leaderboard, df])

        if load_mode == 'all':
            leaderboard.sort_values(by=['model_type', 'metric_value'], ascending=True, inplace=True)
            leaderboard.reset_index(drop=True, inplace=True)
        elif load_mode == 'uniques':
            leaderboard.sort_values(by=['model_type', 'metric_value'], ascending=True, inplace=True)
            leaderboard.drop_duplicates(subset=['model_type'], keep='last', inplace=True)
            leaderboard.sort_values(by=['metric_value'], ascending=True, inplace=True)
            leaderboard.reset_index(drop=True, inplace=True)
        else:
            exit('ERROR: WRONG load_mode. Must be "all" or "uniques" !')

    print('\nModels leaderboard for "{}":\n'.format(target), '-' * 200, '\n',
          leaderboard[['model_type', 'aml_mode', 'features_options', 'metric_type', 'metric_value', 'name']],
          '\n', '-' * 200)
    leaderboard.to_excel('AutoML\leaderboard_' + target + '_' + load_mode + '_mode' + '.xlsx')
    return leaderboard


def model_extract(path, model_type):
    try:                                                                                        # TODO: check the del
        del model
    except NameError:
        pass
    if model_type == 'Xgboost':
        model = xgb.XGBClassifier()
        model.load_model(path)
    elif model_type == 'CatBoost':
        model = CatBoostClassifier()
        model.load_model(path)
    elif model_type == 'Ensemble':
        if path.endswith('Ensemble'):
            path = path.replace('\\Ensemble', '')
        model = AutoML(results_path=path)
    else:
        model = joblib.load(path)
    return model


def load_model(model_info):
    if m.model_type == 'Ensemble':
        model = model_extract(model_info['path'], m.model_type)
        cv_fold_num = ['fold_0', 'fold_1', 'fold_2', 'fold_3', 'fold_4']                      # TODO: get from dir
        return [model, model, model, model, model], cv_fold_num                               # TODO !!! model x folds
    else:
        models = []
        cv_fold_nums = []
        for file in os.listdir(model_info['path']):
            if (file.startswith('learner_fold_') and not \
               file.endswith(('.csv', '.log', '.png', '.txt', '_tree'))) or \
               file.endswith(('.extra_trees', '.decision_tree')):                             # TODO: filelist
                path = model_info['path'] + '\\' + file
                cv_fold_num = re.search(r'fold_\d', file).group(0)
                model = model_extract(path, m.model_type)
                models.append(model)
                cv_fold_nums.append(cv_fold_num)
        return models, cv_fold_nums



if __name__ == '__main__':

    # main params
    targets = ['Исход MACCE КТ']
    aml_folder = 'AutoML'
    aml_modes = ['Perform'] # 'Explain', 'Optuna', 'Compete'

    # load prepared data
    excel_file = 'dataset\prepared_data.xlsx'
    df = pd.read_excel(excel_file, header=0, index_col=0)

    # separating data on DataFrames and set types
    binary_cols = df.loc[:, df.isin([0, 1]).all()]
    df[binary_cols.columns] = binary_cols.astype(np.uint8, copy=False)
    non_binary_cols = df.drop(columns=binary_cols.columns, axis=1, inplace=False)

    # set TARGET and PREDICTORS dataframes
    df_targets = df.loc[:, df.columns.str.contains('^Исход.*') == True]
    df_predictors = df.drop(df_targets.columns, axis=1)

    # targets loop
    for target in targets:
        leaderboard = get_leaderboard(target, aml_folder, aml_modes, 'uniques')
        #leaderboard.drop(leaderboard.tail(1).index, inplace=True)                                                # ! del
        ldb_metrics = pd.DataFrame() # results of the models testing

        # model loop
        for _, m in leaderboard.iterrows():
            print('Target:', target, '| Model:', m.model_type, '| Mode:', m.aml_mode)
            models, cv_folds = load_model(m)
            model_class = models[0].__class__.__name__
            predictions = pd.DataFrame() # models predictions df

            # cv folds loops from automl folds folder
            for i, fold in enumerate(cv_folds):

                # load stored NumPy folds for each model
                print('\t -> Model:', models[i].__class__.__name__, '|', fold, end="\t")
                path = 'AutoML\\AutoML_' + target + '_' + m.aml_mode + '\\folds\\' + fold + '_train_indices.npy'
                i_train = np.load(path) # training indices
                path = 'AutoML\\AutoML_' + target + '_' + m.aml_mode + '\\folds\\' + fold + '_validation_indices.npy'
                i_val = np.load(path) # validation indices

                # load model framework
                path = 'AutoML\\AutoML_' + target + '_' + m.aml_mode + '\\' + m['name'] +'\\framework.json'
                if m.model_type != 'Ensemble': # if NOT AutoML Ensemble model
                    with open(path, 'r') as f:
                        framework = json.load(f)

                    # X scaling
                    X = df_predictors.copy()
                    y = df_targets[target].copy()

                    col_scaled = framework['params']['preprocessing']['columns_preprocessing']
                    col_scaled = [col for col in col_scaled if col_scaled[col] == ['scale_normal']] # TODO add logscaller
                    print('scaled columns', len(col_scaled))
                    if col_scaled != []:
                        scaler = StandardScaler()
                        X[col_scaled] = scaler.fit_transform(X[col_scaled])

                    # X features drop
                    if 'drop_features' in framework['preprocessing'][i]:
                        drop_features = framework['preprocessing'][i]['drop_features']
                        X_val = X.iloc[i_val].drop(drop_features, axis=1)
                        X_train = X.iloc[i_train].drop(drop_features, axis=1)
                        print('Drop ON', end='\t')
                    else:
                        X_val = X.iloc[i_val]
                        X_train = X.iloc[i_train]
                        print('Drop OFF', end='\t')

                    y_val = y.iloc[i_val]

                # in case of AutoML Ensemble model
                else:
                    X = df_predictors.copy()
                    y = df_targets[target].copy()

                    X_val = X.iloc[i_val]
                    y_val = y.iloc[i_val]
                    X_train = X.iloc[i_train]
                    print('AutoML pipeline')


                # testing model in folds loop
                if m.model_type == 'Ensemble':
                    print('!!! AML !!!')
                    path = 'AutoML\\AutoML_' + target + '_' + m.aml_mode + '\\Ensemble\\ensemble.json'
                    with open(path, 'r') as f:
                        json_object = json.load(f)
                    aml_list = json_object['selected_models']

                    models = []
                    for aml in aml_list:
                        print(aml['model'], aml['repeat'], '\n')

                        # print(leaderboard)





                else:
                    y_pred = models[i].predict(X_val)
                    y_pred = pd.DataFrame(data=y_pred, index=y_val.index, columns=['y_pred'])
                    y_prob = models[i].predict_proba(X_val)[:, 1]
                    y_prob = pd.DataFrame(data=y_prob, index=y_val.index, columns=['y_prob'])
                    res = pd.concat([y_val, y_prob, y_pred], axis=1, ignore_index=False, sort=False)
                    res.set_axis(['y_val', 'y_prob', 'y_pred'], axis=1, inplace=True)
                    predictions = pd.concat([predictions, res], axis=0,  sort=False)

            path = 'AutoML\\predictions_out_of_folds_' + target + '_' + m.aml_mode + '_' + m['name'] + '.xlsx'
            predictions.to_excel(path)
                                                                                    # ! del
            metric_roc = roc_auc_score(predictions.y_val, predictions.y_prob, average=None)
            print('Predict_proba for all folds:')
            print('ROC AUC:', metric_roc, '\n')

            ldbs = pd.Series(
                data=[target, m['aml_mode'], m['name'], model_class, m['metric_value'], m['train_time'], metric_roc],
                index=['target', 'mode', 'name', 'model_type', 'old_metric', 'train_time', 'ROC_AUC']
            )

            ldb_metrics = pd.concat([ldb_metrics, ldbs], axis=1, sort=False, ignore_index=True)

        print(ldb_metrics)
        path = 'AutoML\\new_leaderboard_' + target + '.xlsx'
        ldb_metrics.to_excel(path, header=False)









            # if model_name == 'Xgboost':
            #     print('Xgboost!!!')
            # elif model_name == 'CatBoost':
            #     print('CatBoost!!!')
            # else:
            #     rocs = []
            #     y_pred = []
            #     y_prob = []
            #     y_val = []
            #     i_temp=[]
            #     for file in os.listdir(path):
            #         if file.startswith('learner_fold_') and not file.endswith(('.csv', '.log', '.png')):
            #             file = path + '\\' + file
            #             print('File loaded: ', file)
            #             model = joblib.load(file)
            #             # print(model.__class__.__name__)
            #
            #             cv_fold_num = re.search(r'fold_\d', file).group(0)
            #             print(cv_fold_num)
            #
            #             i_train = np.load(path_base + '\\folds\\'
            #                               + cv_fold_num + '_train_indices.npy')
            #             i_val = np.load(path_base + '\\folds\\'
            #                             + cv_fold_num + '_validation_indices.npy')
            #             X_val = X.iloc[i_val]
            #             y_val.extend(y[i_val])
            #
            #             y_pred.extend(model.predict(X_val))
            #             y_prob.extend(model.predict_proba(X_val)[:, 1])
            #
            #             # print('Weights:', model.coefs_)
            #             # print(model.predict_proba(X_val))
            #             rocs.append(roc_auc_score(y[i_val], model.predict_proba(X_val)[:, 1], average=None))
            #             # print(X_val)
            #             i_temp.extend(i_val)
            # pd.DataFrame([i_temp, y_val, y_pred, y_prob], index=['i', 'y_val','predict', 'proba']).transpose().to_excel(path + '\\y.xlsx')
            #
            #
            # print('All_aucs', rocs)
            # print('mean_auc', np.mean(rocs))
            # print('median_auc', np.median(rocs), np.quantile(rocs, 0.25), np.quantile(rocs, 0.75))
            #
            # print('\npredict_proba for all folds:')
            #
            # print('roc_auc', roc_auc_score(y_val, y_prob, average=None))
            # print('roc_auc_micro', roc_auc_score(y_val, y_prob, average='micro'))
            # print('roc_auc_macro', roc_auc_score(y_val, y_prob, average='macro'))

















    # y_pred = model.predict(df_predicts)
    # p1 = model.predict_proba(df_predicts)[:, 1]

    # file = 'AutoML\AutoML_Исход MACCE КТ_Perform\\1_Linear\\framework.json'
    # with open(file, 'r') as f:
    #     model = json.load(f)
    # print(model)
    #

    # y_pred = model.predict(df_predicts)
    # print(y_pred)


    # path = 'AutoML\AutoML_Исход MACCE КТ_Perform'
    # model = AutoML(results_path=path, algorithms="1_Linear")
    #
    #
    # p1 = model.predict_proba(df_predicts)[:, 1]
    # y_pred = model.predict(df_predicts)
    #
    # print('\npredict_proba:')
    #
    # print('roc_auc', roc_auc_score(df_target, p1, average=None))
    # print('roc_auc_micro', roc_auc_score(df_target, p1, average='micro'))
    # print('roc_auc_macro', roc_auc_score(df_target, p1, average='macro'))
    #
    # print('\npredict 1/0:')
    #
    # print('roc_auc', roc_auc_score(df_target, y_pred, average=None))
    # print('roc_auc_micro', roc_auc_score(df_target, y_pred, average='micro'))
    # print('roc_auc_macro', roc_auc_score(df_target, y_pred, average='macro'))
    #
    # fpr, tpr, thresholds = roc_curve(df_target, p1)
    # roc_auc = auc(fpr, tpr)
    #
    # cf_matrix = confusion_matrix(df_target, y_pred)
    #
    # # plt.figure()
    # # group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    # # group_counts = ['{0: 0.0f}'.format(value) for value in cf_matrix.flatten()]
    # # group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    # # labels = [f'{v1} \n{v2} \n{v3}' for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    # # labels = np.asarray(labels).reshape(2, 2)
    # # sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')


















    # plt.figure()
    # lw = 2
    # plt.plot(
    #     fpr,
    #     tpr,
    #     color="darkorange",
    #     lw=lw,
    #     label="ROC curve (area = %0.2f)" % roc_auc,
    # )
    # plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("Receiver operating characteristic example")
    # plt.legend(loc="lower right")


    plt.show()





















    # # load prepared data
    # excel_file = 'dataset\prepared_data.xlsx'
    # df = pd.read_excel(excel_file, header=0, index_col=0)
    #
    # # separating data on DataFrames and set types
    # binary_cols = df.loc[:, df.isin([0, 1]).all()]                                      # select the binary cols only
    # df[binary_cols.columns] = binary_cols.astype(np.uint8, copy=False)                  # predictors binary as uint8
    # non_binary_cols = df.drop(columns=binary_cols.columns, axis=1, inplace=False)       # select non binary cols only
    #
    # df_targets = df.loc[:, df.columns.str.contains('^Исход.*') == True]                 # target DataFrame cols
    # df_predicts = df.drop(df_targets.columns, axis=1)                                   # all predictors DataFrame
    # X = df_predicts
    #
    # file = 'AutoML\AutoML_Исход MACCE КТ_Perform\\19_RandomForest_GoldenFeatures\predictions_out_of_folds.csv'
    #
    #
    # df2 = pd.read_csv(file)
    #
    # df2['pred2'] = [0 if val < 0.561805 else 1 for val in df2['prediction'].values]
    #
    # y_test = df2['target']
    # y_score = df2['pred2']
    #
    # # print(classification_report(df2['pred2'], df2['target']))
    # print('roc_auc', roc_auc_score(df2['target'], df2['pred2'], average=None))
    # print('roc_auc_micro', roc_auc_score(df2['target'], df2['pred2'], average='micro'))
    # print('roc_auc_macro', roc_auc_score(df2['target'], df2['pred2'], average='macro'))
    # print('accuracy', accuracy_score(df2['target'], df2['pred2']))
    # print('f1', f1_score(df2['target'], df2['pred2'], average=None, zero_division=1))
    # print('f1_micro', f1_score(df2['target'], df2['pred2'], average='micro'))
    # print('f1_macro', f1_score(df2['target'], df2['pred2'], average='macro'))
    #
    #
    #
    # precision, recall, thresholds = precision_recall_curve(y_test, y_score)
    #
    # print(precision, recall, thresholds)
    #
    #
    # fpr, tpr, _ = roc_curve(y_test, y_score)
    # roc_auc = auc(fpr, tpr)
    # plt.figure()
    # lw = 2
    # plt.plot(
    #     fpr,
    #     tpr,
    #     color="darkorange",
    #     lw=lw,
    #     label="ROC curve (area = %0.2f)" % roc_auc,
    # )
    # plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("Receiver operating characteristic example")
    # plt.legend(loc="lower right")
    # plt.show()

