"""
    Streamlit app for SaaS growth analysis
    Bulk of the work happens in ranges.py
    Usage: streamlit run app.py

    Analyzing growth for SaaS companies across a few scenarios:
        (1.) Cross-sectional analysis of growth rates in the year leading up to an IPO
                (a.) ARR at IPO (Scatter plot and box plot)
                (b.) Growth rate at IPO (Scatter plot)
                (c.) ARR vs CAGR plots

        (2.) Longitudinal analysis of growth rates indexed to revenue (e.g growth rate from $200-$400M in revenue)
                (a.) Fastest from $Xmm to $Ymm in ARR (Bar chart)
                (b.) Implied CAGR from $X to $Y in ARR (Scatter plot)
                (c.) Estimated CAGR based on given growth endurance

        (3.) 1 yr forward CAGR

    Filter options:
        (a.) Select revenue ranges
        (b.) Company status (Public or Private)

    Author: ramu@acapital.com
    Last updated: 05/09/21
    Dataset: Revenue data for 66 SaaS Companies from a few quarters before IPO to May 2021
"""

import pandas as pd
import streamlit as st
import ranges


def _not_implemented(m_df):
    st.title("Coming soon!")


def ui_toplevel_sidebar_navigation(m_df):
    pages_dispatch = {'Camera-ready': ranges.benchmark_main,
                      'ARR range': ranges.benchmark_main,
                      '1yr CAGR at ARR': ranges.benchmark_main}

    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Benchmarking:", list(pages_dispatch.keys()))
    return pages_dispatch[selection](m_df, selection)


def load_dataset(filename='master.csv'):
    df = pd.read_csv('master.csv')
    df = df.reset_index(drop=True)
    tickers_to_exclude = ['TTD','SYMC', 'GDDY', 'CARB', 'RP', 'NTNX', 'RNG', 'SWI', 'PANW', 'SPLK', 'FEYE', 'PING']
    tickers = list(set(df['ticker']) - set(tickers_to_exclude))
    df = df[df['ticker'].isin(tickers)]
    return df


def main():
    with st.spinner('Loading companies and creating dataset...'):
        m = load_dataset()

    num_companies = m['ticker'].nunique()
    st.success(f'{num_companies} companies loaded')
    ui_toplevel_sidebar_navigation(m)

if __name__ == "__main__":
    main()
