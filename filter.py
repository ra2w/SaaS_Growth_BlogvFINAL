import streamlit as st
import pandas as pd
import numpy as np



def get_optimal_hist_bin_size(metric,df):
    #  Freedman–Diaconis rule
    q1 = df[metric].quantile(0.25)
    q3 = df[metric].quantile(0.75)
    iqr = q3 - q1
    n = df[metric].count()
    h = int(2*iqr/(n**(1.0/3)))
    h = np.maximum(1,h)
    st.write(h)
    return h


def by_gtm(selected_gtm_option,df):
    return df[df['ticker'].isin(schema.tickers_by_gtm(selected_gtm_option))]

def by_time(selected_time,df,range=False):
    if range:
        t0 = selected_time[0]
        t1 = selected_time[1]
        assert(t1>=t0)
        return df[(df['t'] >= t0) & (df['t'] <= t1)]
    else:
        return df[df['t'] == selected_time]



def by_metric_range(metric,df,selected_range):

    low_range , high_range = selected_range

    # STEP 1: for each ticker, figure out the last time period where metric < low_range
    df_start_time = pd.DataFrame(df.where((df[metric] < low_range) & (df[metric] > 0)).groupby('ticker')['t'].max())
    df_start_time.reset_index(inplace=True)


    #STEP 2: for each ticker, figure out the first time period where metric > low_range
    df_end_time = pd.DataFrame(df.where(df[metric] > high_range).groupby('ticker')['t'].min())
    df_end_time.reset_index(inplace=True)

    # STEP 3: for each ticker, get relevant metrics at start_time and end_time
    #         result: dataframe that looks like this: {ticker: 'Start Date','t_x','ARR_x','t_y','ARR_y'}
    df_at_low = pd.merge(df, df_start_time, on=['t', 'ticker'], how='inner')
    df_at_high = pd.merge(df, df_end_time, on=['t', 'ticker'], how='inner')
    df_range = pd.merge(df_at_low, df_at_high, on=['ticker'], how='inner')[
        ['ticker', 'date_x','t_x', metric+'_x', 'date_y','t_y', metric+'_y']]
    df_range = df_range.rename(columns={'date_x':'Start Date','date_y':'End Date',
                                        metric+'_x':metric+'(0)',
                                        metric+'_y':metric+'(1)'})

    # STEP 4: Remove tickers with invalid ranges
    ex_tickers = list(df_range.where(df_range['t_y'] < df_range['t_x']).dropna(axis=0, subset=['ticker'])['ticker'])
    #st.write(f"excluded ticker = {ex_tickers}")
    df_range = df_range[~df_range['ticker'].isin(ex_tickers)]

    df_range = df_range.astype({'t_x': 'int32', 't_y': 'int32'})

    # return: df_range has the following schema: {ticker: 'Start Date','t_x','ARR_x','End Date','t_y','ARR_y'}
    return df_range




