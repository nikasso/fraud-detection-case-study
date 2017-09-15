# Cleaning & Conducting Feature Engineering on Data

import pandas as pd

def quantityTotal(x):
    lst = [i["quantity_total"] for i in x]
    return sum(lst)

def feature_engineering(df):
    '''
    Takes in dataframe read from json file.
    Adds fraud column, creates new dataframe with only relevant features,
    with no null values, and engineers those features.
    Returns new dataframe (so that's ready to be split into X & y data).
    '''

    # Adding fraud column to df
    df['fraud'] = df.acct_type.isin([u'fraudster_event', u'fraudster', \
        u'fraudster_att'])

    # Feature engineering on df
    df= df[df['user_type'].isin([1,3,4,5])]
    df['previous_payouts'] = df['previous_payouts'].apply(lambda x : 1 if len(x)>=1 else 0)
    df["quantity_total"] = df["ticket_types"].apply(quantityTotal)
    df['generic_email']=df['email_domain'].apply(lambda x : 0 if  \
        all(s not in x for s in ['aol', 'gmail', 'live','hotmail','yahoo', \
        'outlook']) else 1)
    df['delivery_method'] = df['delivery_method'].apply(lambda x: 1 if x>=1.0 else 0)
    df = pd.get_dummies(df, columns = ['user_type'])

    # Get sub-selection of columns from df to create new_df
    new_df = df[['generic_email','previous_payouts','delivery_method', \
        'sale_duration2','quantity_total','user_type_1','user_type_3', \
        'user_type_4','user_type_5','fraud']]

    return new_df
