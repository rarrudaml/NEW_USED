###########################
######
###### Clean unused columns and equalize dataset according to traiset
#####
###########################
###### Packages
from datetime import datetime

import pandas as pd

###########################


def step04_clean_dataframes(df_train, df_test):

    df_train.set_index('key', inplace=True)
    df_test.set_index('key', inplace=True)

    df_train = df_train.filter(regex='^(condition|trat_)').fillna(-1)
    df_test = df_test.filter(regex='^(condition|trat_)').fillna(-1)

    # Testset with same columns trainset

    new_cols = set(df_train.drop(columns=['condition']).columns) - set(
        df_test.columns
    )

    # adicionar colunas faltantes em B e preencher com -1
    for col in new_cols:
        df_test[col] = -1

    # remover colunas de B que não estão presentes em A
    for col in df_test.columns:
        if col not in df_train.drop(columns=['condition']).columns:
            df_test.drop(columns=[col], inplace=True)

        # reordenar colunas de B para ficar igual a A
        df_test = df_test[df_train.drop(columns=['condition']).columns].copy()

    return df_train, df_test
