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
    # Feature engineering on df
    df['previous_payouts'] = df['previous_payouts'].apply(lambda x : 1 if len(x)>=1 else 0)
    df["quantity_total"] = df["ticket_types"].apply(quantityTotal)
    df['generic_email']=df['email_domain'].apply(lambda x : 0 if  \
        all(s not in x for s in ['aol', 'gmail', 'live','hotmail','yahoo', \
        'outlook']) else 1)
    df['delivery_method'] = df['delivery_method'].apply(lambda x: 1 if x>=1.0 else 0)
    '''
    user_type_1 = [0 for i in xrange(df.shape[0])]
    user_type_3 = [0 for i in xrange(df.shape[0])]
    user_type_4 = [0 for i in xrange(df.shape[0])]
    user_type_5 = [0 for i in xrange(df.shape[0])]
    '''
    df['user_type_1']=0
    df['user_type_3']=0
    df['user_type_4']=0
    df['user_type_5']=0
    type1=df.iloc[0]["user_type"]
    if type1==1:
        df['user_type_1']=1
    elif type1==3:
        df['user_type_3']=1
    elif type1==4:
        df['user_type_4']=1
    elif type1==5:
        df['user_type_5']=1
    df.pop('user_type')
    ''''
    df = pd.concat([df, pd.Series(user_type_1), pd.Series(user_type_3), pd.Series(user_type_4), pd.Series(user_type_5)])
    for i in xrange(len(df["user_type"])):
        if df.iloc[i]["user_type"] == 1:
            df.iloc[i]["user_type_1"] = 1
        elif df.iloc[i]["user_type"] == 3:
            df.iloc[i]["user_type_3"] = 1
        elif df.iloc[i]["user_type"] == 4:
            df.iloc[i]["user_type_4"] = 1
        else:
            df.iloc[i]["user_type_5"] = 1
    '''
    # Get sub-selection of columns from df to create new_df
    new_df = df[['generic_email','previous_payouts','delivery_method', \
        'sale_duration2','quantity_total','user_type_1','user_type_3', \
        'user_type_4','user_type_5']]
    return new_df
