import pandas as pd
import numpy as np
from pandas import DataFrame

def data_load(
        file: list,
        sheet: list,
        index_col: int,
        header: list
) -> DataFrame:

    # load data
    df = pd.read_excel(file, sheet_name=sheet, header=header, index_col=index_col)

    # remove all df unnamed elements in multiindex
    for i, columns_old in enumerate(df.columns.levels):
        columns_new = np.where(columns_old.str.contains('Unnamed'), '', columns_old)
        df.rename(columns=dict(zip(columns_old, columns_new)), level=i, inplace=True)

    # join columns multi levels
    df.columns = df.columns.map('|'.join).str.strip('|')

    # reset column multi levels
    # columns_new=[]
    # for column in df.columns:
    #     for title in reversed(column):
    #         if title != '':
    #             columns_new.append(title)
    #             break
    # df.columns = columns_new
    # # for duplicated columns
    # s = df.columns.to_series()
    # df.columns = s.add(s.groupby(s).cumcount().astype(str).replace('0', ''))

    return df


if __name__ == '__main__':

    # properties of the loaded data table
    excel_file = 'dataset/dataset.xlsx'
    excel_sheet = 'Sum'
    index_col = 0
    header = [0, 1, 2]

    # data loading
    df = data_load(excel_file, excel_sheet, index_col, header)
    data_list = pd.Series(df.columns)

    # replace string values for num 1|0
    df = df.replace('есть', 1)
    df = df.replace('нет', 0)
    df = df.replace('муж', 1)
    df = df.replace('жен', 0)
    df = df.replace('АКШ', 1)
    df = df.replace('БиМКШ', 0)

    # Change objects ans specific data into categorical data
    # spec_categ_cols = ['Пред. |Оценка|ФК ХСН', 'Пред. |ЭХО-КГ до|ФВ ЛЖ']          # list of extra categorical cols

    categorical_cols = [col for col in df.columns if df[col].dtype == "object"]
    object_nunique = list(map(lambda col: df[col].nunique(), categorical_cols))
    print('The number of unique entries in each column with categorical data:')
    print(pd.Series(dict(zip(categorical_cols, object_nunique))),'\n')

    # categorical_cols.extend(spec_categ_cols)
    # df_categ_labeled = pd.get_dummies(df[categorical_cols])                       # encoding categorical columns
    # df = pd.concat([df, df_categ_labeled], ignore_index=False, axis=1)
    # df = df.drop(categorical_cols, axis=1)

    # find binary data
    binary_cols = df[df.columns].isin([0, np.nan, 1]).all()                         # find NaN|1|0 cols in all data
    binary_cols_lst = list(binary_cols[binary_cols == True].index)                  # list NaN|of 1|0 cols

    # replace NaN to 0 in binary data
    df[binary_cols_lst] = df[binary_cols_lst].fillna(0)

    # data type transformation
    df[binary_cols_lst] = df[binary_cols_lst].astype(np.uint8)                      # uint8 for 1|0 cols
    int64_cols_lst = df.dtypes[df.dtypes == 'int64'].index                          # list of int64 cols
    df[int64_cols_lst] = df[int64_cols_lst].astype(np.float64)                      # int64 -> float64
    print('Checking types of data:')
    print(pd.Series(df.dtypes).groupby(df.dtypes).count())

    # drop imbalanced data outside 10-90%
    binary_disbalace = df[binary_cols_lst]
    binary_disbalace = binary_disbalace[binary_disbalace == 1].count() \
                       / len(binary_disbalace)
    drop_cals = binary_disbalace[(binary_disbalace < 0.10) |
                                 (binary_disbalace > 0.90)].index                   # drop data outside 10-90%
    drop_cals = list(drop_cals)
    drop_cals.remove('Пред. |Общ.|Пол')                                             # save a specific element
    drop_cals.remove('Пред. |ФР ССЗ|АГ')                                            # save a specific element
    df.drop(drop_cals, axis=1, inplace=True)

    # check the remained data
    data_list = pd.DataFrame(data_list, index = data_list.values)
    remained_data = pd.Series('yes', index=[col for col in list(df.columns)
                                            if col in list(data_list.index)])
    data_list.rename(columns={data_list.columns[0]: 'remained'}, inplace=True)
    data_list['remained'] = remained_data
    data_list.to_excel("dataset/remained_data_list.xlsx")

    # fill NA/NaN values in float data
    float_cols_lst = list(df.dtypes[df.dtypes == 'float64'].index)
    df[float_cols_lst] = df[float_cols_lst].fillna(df[float_cols_lst].mean(), axis=0)

    # save prepared data to excel
    df.to_excel("dataset/prepared_data.xlsx", sheet_name='prepared_data')




