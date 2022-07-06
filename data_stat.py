import os

import pandas as pd
import numpy as np
import xlsxwriter
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame
from scipy.stats import fisher_exact

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

if __name__ == '__main__':

    # load prepared data
    excel_file = 'dataset\prepared_data.xlsx'
    df = pd.read_excel(excel_file, header=0, index_col=0)

    # separate data on DataFrames
    df_target = df.loc[:, df.columns.str.contains('^Исход.*') == True]              # target DataFrame
    # df_target = df.loc[:, df.columns.str.contains('Вид.*') == True]

    binary_cols = df[df.columns].isin([0, 1]).all()
    df_binary = df[binary_cols[binary_cols].index]
    df_binary = df_binary.drop(df_target.columns, axis=1)                           # predictors binary DataFrame

    # TODO: add some int to float data
    df_float = df.select_dtypes(include=[float])                                    # predictors float DataFrame
    df_float = df.select_dtypes(include=[int])

    print(pd.Series(df_float.columns))


    # Fisher ODDS ratio calculation
    fisher_ODDS_ratio(df_target, df_binary)

