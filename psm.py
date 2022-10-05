import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymatch.Matcher import Matcher

from psmpy import PsmPy
from psmpy.plotting import *


if __name__ == '__main__':

    # load prepared data
    excel_file = 'dataset\prepared_data.xlsx'
    df = pd.read_excel(excel_file, header=0, index_col=0)

    # separating data on DataFrames and set types
    binary_cols = df.loc[:, df.isin([0, 1]).all()]  # select the binary cols only
    df[binary_cols.columns] = binary_cols.astype(np.uint8, copy=False)  # predictors binary as uint8
    non_binary_cols = df.drop(columns=binary_cols.columns, axis=1, inplace=False)  # select non binary cols only

    df_targets = df.loc[:, df.columns.str.contains('^Исход.*') == True]  # target DataFrame cols
    df_predicts = df.drop(df_targets.columns, axis=1)  # all predictors DataFrame


    # Propensity Score Matching
    X = df_predicts.copy()
    X.reset_index(inplace=True)
    X.to_excel('results\\PSM_X.xlsx')

    psm = PsmPy(X, treatment='Интраоп Вид КШ', indx='Пред.  Общ. ID')
    psm.logistic_ps(balance=False) # for imbalanced data
    psm_pred_data = psm.predicted_data
    psm_pred_data.to_excel('results\\PSM_predicted_data.xlsx')

    psm.knn_matched(matcher='propensity_logit', replacement=False, caliper=6)

    fig = plt.figure(figsize=(5, 5))
    ax = psm.plot_match()
    plt.tight_layout()
    fig.savefig('results\\PSM_plot_match.tiff', dpi=300)


    fig = plt.figure()
    ax = psm.effect_size_plot()
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=8)
    plt.ylabel('')
    plt.tight_layout()
    fig.savefig('results\\PSM_effect_size_plot.tiff', dpi=300)

    effect_size = psm.effect_size
    effect_size.to_excel('results\\PSM_effect_size.xlsx')
    # print('Effect sizes per variable:\n', effect_size)

    Error_before = effect_size[effect_size['matching'] == 'before']['Effect Size'].abs().sum()
    Error_after = effect_size[effect_size['matching'] == 'after']['Effect Size'].abs().sum()
    print('Error_before: {:.02f}\tError_after: {:.02f}'.format(Error_before, Error_after))


    matched_ids = psm.matched_ids
    matched_ids.to_excel('results\\PSM_matched_ids.xlsx')

    df_matched = psm.df_matched
    df_matched.to_excel('results\\PSM_df_matched.xlsx')

    print('{} matched patients from {}'.format(df_matched.shape[0], X.shape[0]))

    # plt.show()