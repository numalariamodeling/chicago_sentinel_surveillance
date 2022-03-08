import pandas as pd
import scipy
import numpy as np

def pearsonr(df1, 
             df2, 
             param='Q0.5', 
             min_date=pd.to_datetime('2020-08-13', utc=True), 
             max_date=pd.to_datetime('2021-06-07', utc=True)):
    '''
    Checks Pearson correlation of two R(t) series between two dates.
    '''
    new_df = pd.merge(df1, df2, left_on=df1.index, right_on=df2.index, how='inner')
    new_df = new_df[(new_df['key_0'] >= min_date) & (new_df['key_0'] <= max_date)]
    return scipy.stats.pearsonr(new_df[param + '_x'], new_df[param + '_y']), len(new_df)

def percent_agreement(df1, 
             df2, 
             param='Q0.5', 
             min_date=pd.to_datetime('2020-08-13', utc=True), 
             max_date=pd.to_datetime('2021-06-07', utc=True)):
    '''
    Checks percent agreement of two R(t) series between two dates.
    '''
    new_df = pd.merge(df1, df2, left_on=df1.index, right_on=df2.index, how='inner')
    new_df = new_df[(new_df['key_0'] >= min_date) & (new_df['key_0'] <= max_date)]
    return np.sum((new_df[param + '_x'] >= 1) == (new_df[param + '_y'] >= 1))/len(new_df)

def spearmanr(df1, 
             df2, 
             param='Q0.5', 
             min_date=pd.to_datetime('2020-08-13', utc=True), 
             max_date=pd.to_datetime('2021-06-07', utc=True)):
    '''
    Checks Spearman correlation of two R(t) series between two dates.
    '''
    new_df = pd.merge(df1, df2, left_on=df1.index, right_on=df2.index, how='inner')
    new_df = new_df[(new_df['key_0'] >= min_date) & (new_df['key_0'] <= max_date)]
    return scipy.stats.spearmanr(new_df[param + '_x'], new_df[param + '_y']), len(new_df)

def med_and_iqr(discrete_distrb):
    '''
    Returns 50th, 25th and 75th percentiles of a discrete distribution
    represented by an array containing the probability density function.
    '''
    p25 = np.arange(len(discrete_distrb))[discrete_distrb.cumsum() >= 0.25][0]
    p75 = np.arange(len(discrete_distrb))[discrete_distrb.cumsum() >= 0.75][0]
    p50 = np.arange(len(discrete_distrb))[discrete_distrb.cumsum() >= 0.50][0]
    return p50, (p25, p75)

def zip_in_zips(zip_code, zips):
    '''
    Checks if ZIP code is in a list of provided ZIP codes.
    '''
    try:
        if '-' in str(zip_code):
            return_val = float(zip_code[0:5]) in zips
        else:
            return_val = float(zip_code) in zips
    except:
        return_val = False
    return return_val