import os
import numpy as np
import pandas as pd
from pandas import DataFrame

import xlsxwriter
import matplotlib.pyplot as plt
import seaborn as sns

import scipy
from scipy.stats import fisher_exact
from sklearn.feature_selection import mutual_info_regression


import warnings
warnings.filterwarnings("ignore")

path = 'results'
os.makedirs(path, exist_ok=True)

# Fisher ODDS ratio function
def fisher_ODDS_ratio(
        df_target: DataFrame,
        df_predicts: DataFrame
) -> int:
    fisher = pd.DataFrame(columns=['Target', 'Predictor', 'ODDS_ratio',
                                   'p-value', '95%_CI_lower', '95%_CI_upper'])      # init empty table
    # predict-target combinations loop
    for t_col in df_target.columns:
        for p_col in df_predicts.columns:
            predicts = df_predicts[p_col]
            targets = df_target[t_col]

            p0t0 = ((predicts == 0) & (targets == 0)).sum()                         # contingency tables
            p0t1 = ((predicts == 0) & (targets == 1)).sum()
            p1t0 = ((predicts == 1) & (targets == 0)).sum()
            p1t1 = ((predicts == 1) & (targets == 1)).sum()
            table = np.array([[p0t0, p0t1], [p1t0, p1t1]])
            LOR = np.log(p0t0) + np.log(p1t1) - np.log(p0t1) - np.log(p1t0)         # log odds ratio
            SE = np.sqrt(((1 / table).sum()))                                       # standard error
            CI_lw = round(np.exp(LOR - 1.96 * SE), 3)                               # 95% confidence intervals (lower)
            CI_up = round(np.exp(LOR + 1.96 * SE), 3)                               # 95% confidence intervals (upper)
            ODDS_ratio, p_value = fisher_exact(table)

            result = pd.DataFrame(  # result dataframe
                [[targets.name, predicts.name, ODDS_ratio, p_value, CI_lw, CI_up]],
                columns=['Target', 'Predictor', 'ODDS_ratio', 'p-value', '95%_CI_lower', '95%_CI_upper'],
            )
            fisher = pd.concat([fisher, result], axis=0, ignore_index=True)  # add result to fisher dataframe
    # export results to excel:
    with pd.ExcelWriter('results\stat.xlsx', engine='xlsxwriter') as writer:
        fisher.to_excel(writer, sheet_name='ODDS ratio', startcol=0, startrow=0)
        workbook = writer.book
        sheet = writer.sheets['ODDS ratio']

        format_000 = workbook.add_format({'num_format': '0.000'})
        red_format = workbook.add_format({'bg_color': '#FFC7CE'})
        grn_format = workbook.add_format({'bg_color': '#C6EFCE'})

        sheet.set_column('B:C', 35)
        sheet.set_column('D:G', 15, format_000)

        frange = str('E2:E' + str(len(fisher) + 1))
        sheet.conditional_format(frange,
                                 {'type': 'cell',
                                  'criteria': '<=',
                                  'value': 0.05,
                                  'format': grn_format})
    return(0)

def mutual_information(X, y):
    X = X.copy()
    # for colname in X.select_dtypes(["object", "category"]):
    #     X[colname], _ = X[colname].factorize()
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

def cor_plot(df):  # Correlation visualization function
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), 1)                               #  mask for the upper triangle
    fig, ax = plt.subplots(figsize=(16, 8))
    cmap = sns.diverging_palette(240, 10, n=9, as_cmap=True)                        # colormap
    sns.heatmap(corr, mask=mask, cmap=cmap, annot=True,
                vmax=1.0, vmin=-1.0, center=0, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.5},
                annot_kws={"size": 20 / np.sqrt(len(corr))})
    plt.tight_layout()
    fig.savefig('Results\CorrelationMatrix.tiff', dpi=300, format='tif')
    return fig

if __name__ == '__main__':

    # load prepared data
    excel_file = 'dataset\prepared_data.xlsx'
    df = pd.read_excel(excel_file, header=0, index_col=0)

    # separating data on DataFrames
    df_target = df.loc[:, df.columns.str.contains('^Исход.*') == True]              # targets DataFrame
    # df_target = df.loc[:, df.columns.str.contains('Вид.*') == True]

    binary_cols = df.isin([0, 1]).all()                                             # predictors binary DataFrame
    df_binary = df[binary_cols[binary_cols].index]
    df_binary = df_binary.drop(df_target.columns, axis=1)

    float_mask = df.isin([0, 1]).all() == False                                     # predictors float DataFrame
    df_float = df[float_mask[float_mask].index].astype(np.float64)

    df_predicts = df.drop(df_target.columns, axis=1)                                # all predictors DataFrame

    # Fisher ODDS ratio calculation
    fisher_ODDS_ratio(df_target, df_binary)

    # Mutual information data scores
    for target in df_target:
        print('\nScore:', target)
        mi_scores = mutual_information(df_predicts, df_target[target])
        print(mi_scores.head(5))
        
    # Correlation check
    cor_plot(df_float)

    # TODO: add plot for mutual_information
    # TODO: PSM https://towardsdatascience.com/psmpy-propensity-score-matching-in-python-a3e0cd4d2631
    #           https://analyticsmayhem.com/digital-analytics/propensity-score-matching-python/


