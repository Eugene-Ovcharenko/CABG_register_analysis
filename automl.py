import os
import numpy as np
import pandas as pd
from supervised import AutoML

if __name__ == '__main__':

    # create folder
    path = 'AutoML'
    os.makedirs(path, exist_ok=True)

    # load prepared data
    excel_file = 'dataset\prepared_data.xlsx'
    df = pd.read_excel(excel_file, header=0, index_col=0)

    # separating data on DataFrames and set types
    binary_cols = df.loc[:, df.isin([0, 1]).all()]                                      # select the binary cols only
    df[binary_cols.columns] = binary_cols.astype(np.uint8, copy=False)                  # predictors binary as uint8
    non_binary_cols = df.drop(columns=binary_cols.columns, axis=1, inplace=False)       # select non binary cols only

    df_targets = df.loc[:, df.columns.str.contains('^Исход.*') == True]                 # target DataFrame cols
    df_predicts = df.drop(df_targets.columns, axis=1)                                   # all predictors DataFrame
    X = df_predicts

    automl_modes = ['Explain']
    for automl_mode in automl_modes:
        for target in df_targets:
            print('Target:', target)
            y = df_targets[target]
            automl = AutoML(results_path=f"AutoML\AutoML_{target}_{automl_mode}",
                            mode=automl_mode,
                            features_selection=True,
                            explain_level=2,
                            eval_metric='auc',
                            ml_task= 'binary_classification',
                            validation_strategy={'validation_type': 'split',
                                                 'train_ratio': 0.80,
                                                 'shuffle': True,
                                                 'stratify': True
                                                }
                            )
            automl.fit(X, y)

    automl_modes = ['Compete', 'Perform']
    for automl_mode in automl_modes:
        for target in df_targets:
            print('Target:', target)
            y = df_targets[target]
            automl = AutoML(results_path=f"AutoML\AutoML_{target}_{automl_mode}",
                            mode=automl_mode,
                            features_selection=True,
                            explain_level=2,
                            eval_metric='auc',
                            ml_task='binary_classification',
                            validation_strategy={'validation_type': 'kfold',
                                                 'k_folds': 5,
                                                 'shuffle': True,
                                                 'stratify': True,
                                                 'random_seed': 123
                                                 }
                            )
            automl.fit(X, y)

    for target in df_targets:
        print('Target:', target)
        y = df_targets[target]
        automl = AutoML(results_path=f"AutoML\AutoML_{target}_Optuna",
                        mode='Optuna',
                        optuna_time_budget=10000,
                        features_selection=True,
                        explain_level=2,
                        eval_metric='auc',
                        ml_task= 'binary_classification',
                        validation_strategy={'validation_type': 'kfold',
                                             'k_folds': 5,
                                             'shuffle': True,
                                             'stratify': True,
                                             'random_seed': 123
                                             }
                        # validation_strategy={'validation_type': 'split',
                        #                      'train_ratio': 0.80,
                        #                      'shuffle': True,
                        #                      'stratify': True
                        #                      }
                        )
        automl.fit(X, y)

