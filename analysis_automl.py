import os
import warnings
import pandas as pd
import numpy as np
import re
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Union, List, Literal
from supervised import AutoML
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score, accuracy_score, precision_score, average_precision_score, recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# style settings
warnings.filterwarnings("ignore")
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_colwidth', None)
# plt.style.use('seaborn')
# sns.set_context("talk")  # \ "paper" \ "poster" \ "notebook"
# sns.set_style("whitegrid")
sns.set(font_scale=1)
sns.set_style('white', {'xtick.bottom': True, 'xtick.top': False, 'ytick.left': True, 'ytick.right': False,
                        'axes.spines.left': True, 'axes.spines.bottom': True, 'axes.spines.right': True,
                        'axes.spines.top': True, 'font.family': 'sans serif', 'font.sans-serif': 'Arial',
                        'font.style': 'bold'})
cmap = plt.cm.get_cmap('coolwarm') # YlGnBu


def get_leaderboard(
        target: str,
        aml_folder: str,
        aml_modes: str,
        load_mode: Literal["all", "uniques"] = "uniques"
) -> pd.DataFrame:
    '''
    Read the AutoML mljar-supervised results function.
    In current version Golden Features, Stacked and Essembly models drop !!!

    :param target: name of AutoML subfolder for target,
    :param aml_folder: name of folder of AutoML results,
    :param aml_modes: 'Perform', 'Optuna', 'Compete',
    :param load_mode: 'all' (all best models), "uniques" (only one model for each class).
    :return: leaderboard models DataFrame
    '''
    leaderboard = pd.DataFrame()
    for mode in aml_modes:
        path = aml_folder + '\AutoML_' + target + '_' + mode
        df = pd.read_csv(path + '\leaderboard.csv')
        df.sort_values(by=['model_type', 'metric_value'], ascending=False, inplace=True)

        # DROP Golden Features
        df['features_options'] = df['name'].apply(
            lambda n: re.findall('GoldenFeatures', n)[0] if re.findall('GoldenFeatures', n) else "")
        df.drop(df[df['features_options'] == 'GoldenFeatures'].index, inplace=True)

        # DROP Stacked models
        df['features_options'] = df['name'].apply(
            lambda n: re.findall('Stacked', n)[0] if re.findall('Stacked', n) else "")
        df.drop(df[df['features_options'] == 'Stacked'].index, inplace=True)

        # DROP Ensemble models
        df.drop(df[df['model_type'] == 'Ensemble'].index, inplace=True)

        # DROP Duplicates and sort data
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
    leaderboard.to_excel('AutoML\\leaderboard_' + target + '_' + load_mode + '_mode' + '.xlsx')
    return leaderboard


def model_extractor(
        path: str
) -> list:
    '''
    Extracting fitted models from the given path for each cv_fold

    :param path: path of the model to extract,
    :return: model list for each cv_fold
    '''

    models = []
    for file in os.listdir(path):
        if file.startswith('learner_fold_') and \
                file.endswith(('.baseline', '.linear', '.k_neighbors', '.decision_tree', '.extra_trees',
                               '.random_forest', '.xgboost', '.catboost', '.lightgbm', '.neural_network')):
            f_path = path + '\\' + file

            if file.endswith('.xgboost'):
                model = XGBClassifier()
                model.load_model(f_path)
            elif file.endswith('.catboost'):
                model = CatBoostClassifier()
                model.load_model(f_path)
            elif file.endswith('.lightgbm'):
                model = LGBMClassifier()
                model.load_model(f_path)
            else:
                model = joblib.load(f_path)
            models.append(model)

    return models


def cv_folds_extractor(
        path: str
) -> pd.DataFrame:
    '''
    Extracting and return validation/train cv_folds from the path

    :param path: path of the model to extract,
    :return: model list for each cv_fold
    '''

    path = os.path.join(path, 'folds')
    cv_folds = pd.DataFrame()

    for file in os.listdir(path):
        if file.startswith('fold_') and file.endswith('_validation_indices.npy'):
            cv_fold_num = re.search(r'fold_\d', file).group(0)
            f_path = os.path.join(path, file)
            idx = np.load(f_path)  # validation indices
            cv_folds = cv_folds.append(pd.Series(data=idx, name=('validation_' + cv_fold_num)))

        elif file.startswith('fold_') and file.endswith('_train_indices.npy'):
            cv_fold_num = re.search(r'fold_\d', file).group(0)
            f_path = os.path.join(path, file)
            idx = np.load(f_path)  # validation indices
            cv_folds = cv_folds.append(pd.Series(data=idx, name=('train_' + cv_fold_num)))

    return cv_folds


def data_preprocessing(
        path: str,
        X: pd.DataFrame
)-> pd.DataFrame:
    '''
    Function read the AutoML MLJAR framework and preprocess the data

    :param path: path of the MLJAR framework.json
    :return: Scaled X DataFrame
    '''

    with open(path, 'r') as f:
        framework = json.load(f)

    # columns scale
    col_scaled = framework['params']['preprocessing']['columns_preprocessing']
    col_scaled = [col for col in col_scaled if col_scaled[col] == ['scale_normal']]                                     # TODO: add logscaller !
    print('X scaled column: {} \ {}'.format(len(col_scaled), len(X.columns)), end='\t')
    if col_scaled != []:
        scaler = StandardScaler()
        X[col_scaled] = scaler.fit_transform(X[col_scaled])

    # drop features
    if 'drop_features' in framework['preprocessing'][0]:                                                                # TODO: must check !
        drop_features = framework['preprocessing'][0]['drop_features']
        X.drop(drop_features, axis=1, inplace=True)
        print('dropped features: {} \ {}'.format(len(drop_features), len(X.columns)))
    else:
        print('| No dropped features found')

    return X


def classifier_evaluation(
        y: pd.Series,
        y_pred: pd.Series,
        y_prob: pd.Series
) -> pd.DataFrame:
    """
    Evaluate the performance of the classifier by standard metrics

    :param y: Ground Truth y
    :param y_pred: predicted values by X data
    :param y_prob: probability of predictions by X data
    :return: DataFrame of classifier evaluation metrics
    """
    metric_roc = roc_auc_score(y, y_prob, average=None)
    metric_acc = accuracy_score(y, y_pred)
    metric_prec = precision_score(y, y_pred)
    metric_prec_ave = average_precision_score(y, y_pred)
    metric_rec = recall_score(y, y_pred)
    metric_f1 = f1_score(y, y_pred)

    metrics = pd.Series({
        'ROC_AUC': metric_roc,
        'Accuracy': metric_acc,
        'Precision': metric_prec,
        'Average_Precision': metric_prec_ave,
        'Recall': metric_rec,
        'F1_score': metric_f1
    })
    return metrics


def custom_confusion_matrix(
        y: pd.Series,
        y_pred: pd.Series,
        path: str
) -> None:
    """
    Custom confusion matrix for the given y_Ground_Truth and y_predicted

    :param y_pred: predicted values by X data
    :param y_prob: probability of predictions by X data
    :param path: path with file_name for the figure
    """
    cf_matrix = confusion_matrix(y, y_pred)
    fig = plt.figure()
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ['{0: 0.0f}'.format(value) for value in cf_matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f'{v1} \n{v2} \n{v3}' for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap=cmap)
    plt.tight_layout()
    fig.savefig(path, dpi=300, format='tif')


def multi_roc_curves(
        roc_ftpr: pd.DataFrame,
        path: str,
        format: Literal['dot', 'rus'] = 'dot'
) -> None:
    """
    Plot ROC-curves for all models for one target

    :param roc_ftpr: True & False Positive Rate for al models
    :param path: the directory for images save
    :param format: 'dot' - 0.0 value format, 'rus' - 0,0 value format
    """
    fig = plt.figure(figsize=(7, 6))
    if format == 'dot':
        for i, roc in roc_ftpr.iterrows():
            plt.plot(
                roc.fpr,
                roc.tpr,
                color=cmap(i / len(roc_ftpr)),
                lw=2,
                label=roc.model + " (AUC = %0.2f)" % roc.roc_score
            )
    elif format == 'rus':
        for i, roc in roc_ftpr.iterrows():
            plt.plot(
                roc.fpr,
                roc.tpr,
                color=cmap(i / len(roc_ftpr)),
                lw=2,
                label='{} (AUC={})'.format(roc.model, str(round(roc.roc_score, 3)).replace('.', ','))
            )
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("False Positive Rate", fontsize=18)
    plt.ylabel("True Positive Rate", fontsize=18)
    plt.title("ROC " + target, fontsize=18)
    plt.legend(loc="lower right")  # prop={'size': 11}

    if format == 'rus':
        ax = plt.gca()
        xticks = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], len(ax.get_xticklabels()))
        xticks = [str(round(x, 2)).replace('.', ',') for x in xticks]
        ax.set_xticklabels(xticks)
        yticks = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], len(ax.get_yticklabels()))
        yticks = [str(round(y, 2)).replace('.', ',') for y in yticks]
        ax.set_yticklabels(yticks)

    plt.tight_layout()
    file = 'roc_curves_' + roc_ftpr.target[0] + '.tiff'
    path = os.path.join(path, file)
    fig.savefig(path, dpi=300, format='tif')


if __name__ == '__main__':

    # main params
    aml_folder = 'AutoML'
    aml_modes = ['Perform', 'Compete', 'Optuna']

    # make folders for results store
    path = 'AutoML\Analyzed\predictions'
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    path = 'AutoML\Analyzed\charts'
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

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

    targets = df_targets.columns
    # targets = ['Исход MACCE КТ']

    # targets loop
    for target in targets:
        leaderboard = get_leaderboard(target, aml_folder, aml_modes, 'uniques')
        ldb_metrics = pd.DataFrame() # results of the models testing
        roc_ftpr = pd.DataFrame() # df for roc curves dumping

        # model loop
        for _, m in leaderboard.iterrows():
            print('Target:', target, '| Model:', m.model_type, '| Mode:', m.aml_mode)
            models = model_extractor(m.path)
            model_class = models[0].__class__.__name__
            predictions = pd.DataFrame()  # models predictions df

            # data preprocessing
            X = df_predictors.copy()
            y = df_targets[target].copy()
            path = 'AutoML\\AutoML_' + target + '_' + m['aml_mode'] + '\\' + m['name'] + '\\framework.json'
            X = data_preprocessing(path, X)

            # load stored NumPy folds for each model
            path = 'AutoML\\AutoML_' + target + '_' + m.aml_mode
            folds = cv_folds_extractor(path)
            folds = folds[folds.index.str.contains('validation_fold_')].dropna(axis=1, how='all')

            # cv folds loops from automl folds folder
            predictions = pd.DataFrame()
            print('Fold #:', end='\t')
            for idx, fold in folds.iterrows():
                i_val = fold.dropna().astype(int).values
                i_fold = int(re.search('validation_fold_(\d+)', idx).group(1))
                # print('\t -> Model:', models[i_fold].__class__.__name__, '|', idx)
                print(i_fold, end='-')

                X_val = X.iloc[i_val]
                y_val = y.iloc[i_val]

                if m.model_type != 'Ensemble':
                    y_pred = models[i_fold].predict(X_val)
                    y_pred = pd.DataFrame(data=y_pred, index=y_val.index, columns=['y_pred'])
                    y_prob = models[i_fold].predict_proba(X_val)[:, 1]
                    y_prob = pd.DataFrame(data=y_prob, index=y_val.index, columns=['y_prob'])
                    res = pd.concat([y_val, y_prob, y_pred], axis=1, ignore_index=False, sort=False)
                    res.set_axis(['y_val', 'y_prob', 'y_pred'], axis=1, inplace=True)
                    predictions = pd.concat([predictions, res], axis=0, sort=False)

            # dump the predictions and probabilities
            print('OK!\n')
            path = os.path.join('AutoML', 'Analyzed', 'predictions')
            file = 'predictions_out_of_folds_' + target + '_' + m.aml_mode + '_' + m['name'] + '.xlsx'
            path = os.path.join(path, file)
            predictions.to_excel(path)

            # confusion matrix
            path = os.path.join('AutoML', 'Analyzed', 'charts')
            file = 'confusion_matrix_' + target + '_' + m.aml_mode + '_' + m['name'] + '.tiff'
            path = os.path.join(path, file)
            custom_confusion_matrix(predictions.y_val, predictions.y_pred, path)

            # model evaluation
            metriсs = classifier_evaluation(predictions.y_val, predictions.y_pred, predictions.y_prob)
            desc = pd.Series({
                'target': target,
                'mode': m['aml_mode'],
                'name': m['name'],
                'model_type': model_class,
                'AUTOML metric': m['metric_value'],
                'train_time': m['train_time']
            })
            ldb = pd.concat([desc, metriсs], axis=0, sort=False)
            ldb_metrics = pd.concat([ldb_metrics, ldb], axis=1, sort=False, ignore_index=True)

            # ROC evaluation
            fpr, tpr, thresholds = roc_curve(predictions.y_val, predictions.y_prob)
            roc_ftpr = roc_ftpr.append({
                'target': target,
                'model': model_class,
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds,
                'roc_score': metriсs['ROC_AUC']
            }, ignore_index=True)

        # export results
        print(ldb_metrics)
        path = 'AutoML\Analyzed\Analysis_' + target + '.xlsx'
        ldb_metrics.to_excel(path, header=False)

        # ROC curves dump & plot
        path = 'AutoML\Analyzed\predictions\\roc_ftpr_' + target + '_' + m.aml_mode + '_' + m['name'] + '.xlsx'
        roc_ftpr.to_excel(path)
        path = os.path.join('AutoML', 'Analyzed', 'charts')
        multi_roc_curves(roc_ftpr, path=path, format='rus')








            # -----------------------------------
            #     # testing model in folds loop
            #     if m.model_type == 'Ensemble':
            #         print('!!! AML !!!')
            #         path = 'AutoML\\AutoML_' + target + '_' + m.aml_mode + '\\Ensemble\\ensemble.json'
            #         with open(path, 'r') as f:
            #             json_object = json.load(f)
            #         aml_list = json_object['selected_models']
            #
            #         models = []
            #         for aml in aml_list:
            #             print(aml['model'], aml['repeat'], '\n')
            #
            #             # print(leaderboard)


