# Code for oversampling minority class

import pandas as pd

def oversampling(df):
    '''
    Takes in dataframe with imbalanced classes of fraud True & False and
    returns new dataframe with balanced classes, by oversampling the fraud
    True class.
    '''
    NotFraudCount = len(df.loc[df['fraud']==0])
    FraudCount = len(df.loc[df['fraud']==1])
    ToGo = NotFraudCount-FraudCount

    sampledDf=df[df['fraud']==1].sample(ToGo, replace=True)
    finalDf = pd.concat([df,sampledDf],ignore_index=True)

    return finalDf
