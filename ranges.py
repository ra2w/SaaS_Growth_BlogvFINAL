"""
    Ranges.py
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
"""

import functools
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st

# local imports
import finance as fin

# Available templates  ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"]
px_template = "plotly_dark"

unit = 5


@st.cache(suppress_st_warning=True)
def create_metric_markers_index(df, metric='ARR'):
    if metric != 'ARR':
        raise NotImplementedError

    m0 = metric + '_lo'
    m1 = metric + '_hi'
    df[m1] = np.floor(df['ARR'] / unit) * unit
    df[m0] = np.floor((df['ARR_p'] + unit) / unit) * unit

    def create_interval(row):
        if np.isnan(row['ARR_p']) or \
                np.isnan(row['ARR']): #or row[m0] > row[m1]:
            return list(range(0, 0))

        if row[m1] > row[m0]:
            return list(range(int(row[m0]), int(row[m1]) + unit, unit))
        else:
            return list(range(int(row[m1]), int(row[m0]) + unit, unit))

    df['arr_marker'] = df.apply(create_interval, axis=1).astype('object')
    return df.explode('arr_marker').groupby(['ticker', 'arr_marker'])['row_num'].max().unstack(0)


def marker_to_row_num(metric_name, ticker, df_metric_index, metric_value):
    if metric_name != 'ARR':
        raise NotImplementedError

    if metric_value not in df_metric_index.index:
        return np.NaN

    return df_metric_index.loc[metric_value, ticker]


arr_marker_to_row_num = functools.partial(marker_to_row_num, 'ARR')


def growth_endurance_plot(df_r):
    st.write("Growth analysis")
    st.table(df_r.pivot(index='ticker', columns='ARR range', values='g_e'))

    df_stats = df_r.groupby('ARR range', sort=False)['g_e'].describe().reset_index()
    # st.table(df_stats.melt(id_vars='ARR range', var_name='Legend', value_name='50%'))
    df_plot = df_stats.melt(id_vars='ARR range', var_name='Legend', value_name='50%')

    fig = px.line(df_stats, x='ARR range', y='50%', template=px_template)
    st.plotly_chart(fig)

    with st.beta_expander('Summary Statistics'):
        st.table(df_stats)

def cagr_between_markers(ticker, df_c, arr_ix0, arr_ix1, df_metric_index):
    r0_num = arr_marker_to_row_num(ticker, df_metric_index, arr_ix0)
    r1_num = arr_marker_to_row_num(ticker, df_metric_index, arr_ix1)

    if pd.isna(r0_num) or pd.isna(r1_num):
        return (np.NaN, np.NaN, np.NaN)

    row0 = df_c.loc[(ticker, r0_num), :]
    row1 = df_c.loc[(ticker, r1_num), :]

    r0 = fin.cagr(end_arr=row0['ARR'], start_arr=row0['ARR_p'], t_in_q=row0['t'] - row0['t_p'])
    t_ix0 = row0['t'] - fin.time_between_arr_ranges(end_arr=row0['ARR'], start_arr=arr_ix0, r=r0)
    # st.write(f"arr_ix0 = ${arr_ix0} in interval [{row0['ARR'],row0['ARR_p']}]  r0 = {r0} t_ix0 = {t_ix0}")

    r1 = fin.cagr(end_arr=row1['ARR'], start_arr=row1['ARR_p'], t_in_q=row1['t'] - row1['t_p'])
    t_ix1 = row1['t'] - fin.time_between_arr_ranges(end_arr=row1['ARR'], start_arr=arr_ix1, r=r1)
    # st.write(f"arr_ix1 = ${arr_ix1} in interval [{row1['ARR'], row1['ARR_p']}]  r1 = {r1} t_ix1 = {t_ix1}")

    range_cagr = fin.cagr(arr_ix1, arr_ix0, t_ix1 - t_ix0)
    time_in_yrs = (t_ix1 - t_ix0) / 4.0
    row_time = (row1['t'] - row0['t']) / 4.0
    # st.write(f"{arr_ix0:.2f} [{t_ix0:.2f}] --> {arr_ix1:.2f} [{t_ix1:.2f}] cagr = {range_cagr:.2f}%, years = {time_in_yrs:.2f}")
    return (range_cagr, time_in_yrs, row_time)


def cagr_over_marker_series(df_c, marker_ranges, df_marker_index, short_circuit):
    c0, c1, c2, c3 = ([], [], [], [])
    ticker = df_c.name
    for i in marker_ranges:
        (cagr, time_in_yrs, row_time) = cagr_between_markers(ticker, df_c, i.left, i.right, df_marker_index)
        if not np.isnan(cagr):
            c0.append(i)
            c1.append(cagr)
            c2.append(time_in_yrs)
            c3.append(row_time)
            if short_circuit:
                break
    rdf = pd.DataFrame.from_dict(data={'ARR range': c0, 'CAGR': c1, 'Time (Yrs)': c2, 'Row Time (Yrs)': c3})
    rdf['ARR range'] = rdf['ARR range'].astype('object')
    rdf = rdf.dropna()
    return rdf


def cagr_one_year_forward(df_c, df_marker_index, arr):
    ticker = df_c.name
    r0_num = arr_marker_to_row_num(ticker, df_marker_index, arr)
    if pd.isna(r0_num):
        return pd.DataFrame.from_dict(data={'ARR range': [],
                                            'CAGR': [],
                                            'Time (Yrs)': []})

    row0 = df_c.loc[(ticker, r0_num), :]
    r0 = fin.cagr(end_arr=row0['ARR'], start_arr=row0['ARR_p'], t_in_q=row0['t'] - row0['t_p'])
    t_ix0 = row0['t'] - fin.time_between_arr_ranges(end_arr=row0['ARR'], start_arr=arr, r=r0)
    t_ix1 = row0['t_1yr_f']
    one_year_cagr = fin.cagr(row0['ARR_1yr_f'], arr, t_ix1 - t_ix0)
    time_in_yrs = (t_ix1 - t_ix0) / 4.0
    rdf = pd.DataFrame.from_dict(data={'ARR range': [arr],
                                       'CAGR': [one_year_cagr],
                                       'Time (Yrs)': [time_in_yrs]})
    rdf = rdf.dropna()
    return rdf


def cohort_cagr_one_year_forward(df_marker_index, df, arr):
    """
        Compute 1yr forward CAGR
        Input:
            (1.) Master Dataframe
            (2.) Marker index dataframe: ticker + arr --> row number

        Output:
            Custom dataframe with markers:
                Ticker --> ARR range, CAGR, Time (Yrs)
    """
    rdf = df.groupby('ticker') \
        .apply(cagr_one_year_forward,
               df_marker_index=df_marker_index,
               arr=arr)
    rdf['Growth multiple'] = rdf['CAGR'] / 100 + 1
    return rdf



def cohort_cagr_over_marker_series(df, df_marker_index, interval_ranges, short_circuit=False):
    """
    Input:
        (1.) Master Dataframe
        (2.) Marker index dataframe: ticker + arr --> row number

    Output:
        Custom dataframe with markers:
            Ticker --> ARR range, CAGR, ARR_lo, ARR_hi, CAGR_prev, growth enduranace
    """

    rdf = df.groupby('ticker') \
        .apply(cagr_over_marker_series,
               marker_ranges=interval_ranges,
               df_marker_index=df_marker_index,
               short_circuit=short_circuit)
    rdf['ARR_r_lo'] = rdf['ARR range'].apply(lambda x: x.left)
    rdf['ARR_r_hi'] = rdf['ARR range'].apply(lambda x: x.right)
    rdf['ARR range'] = rdf['ARR range'].astype(str)
    rdf['CAGR_p'] = rdf.groupby('ticker')[['CAGR']].shift(1)
    rdf['Growth multiple'] = rdf['CAGR'] / 100 + 1
    return rdf


def compute_cohort_cagr_panels(df_marker_index, df):
    ex1 = st.beta_expander("Individual companies")
    ex1.table(df.reset_index())

    interval_ranges = pd.IntervalIndex.from_tuples(
        [(40, 75), (50, 100), (60, 100), (75, 150), (150, 300), (200, 300), (200, 400)])
    col_labels = [str(x) for x in interval_ranges]
    df_r = cohort_cagr_over_marker_series(df=df, df_marker_index=df_marker_index, interval_ranges=interval_ranges,
                                          short_circuit=False)

    # panel by growth
    ex2 = st.beta_expander("Panel data (growth)")
    df_rg = df_r.reset_index(level=1, drop=True)
    df_rg = df_rg[['ARR range', 'Growth multiple']]
    df_rg = df_rg.reset_index()
    df_rg = df_rg.pivot(index='ticker', columns='ARR range', values='Growth multiple')
    df_rg = df_rg[[x for x in col_labels if x in df_rg.columns]]

    df_rg['g_e_50_150'] = df_rg['(150, 300]'] / df_rg['(50, 100]']
    df_rg['g_e_50_200'] = df_rg['(200, 400]'] / df_rg['(50, 100]']
    g_e = df_rg['g_e_50_150'].median()

    ex2.table(df_rg)
    ex2.table(df_rg['g_e_50_150'].describe())

    # panel by time
    ex3 = st.beta_expander("Panel data (time)")
    df_rt = df_r.reset_index(level=1, drop=True)
    df_rt = df_rt[['ARR range', 'Time (Yrs)']]
    df_rt = df_rt.reset_index()
    df_rt = df_rt.pivot(index='ticker', columns='ARR range', values='Time (Yrs)')
    df_rt = df_rt[[x for x in col_labels if x in df_rt.columns]]

    df_rt['t_50_150'] = df_rt['(150, 300]'] / df_rt['(50, 100]']
    df_rt['t_50_200'] = df_rt['(200, 400]'] / df_rt['(50, 100]']
    t_d = df_rt['t_50_150'].quantile(0.50)

    ex3.table(df_rt)
    ex3.table(df_rt['t_50_150'].describe())

    fig = px.scatter(df_rt, x='(50, 100]', y='(150, 300]', template=px_template)
    ex3.plotly_chart(fig)

    return t_d


def display_cohort_cagr_one_year_forward(df_marker_index, df, arr):
    df_r = cohort_cagr_one_year_forward(df=df, arr=arr, df_marker_index=df_marker_index)
    st.markdown("---")
    display_cohort_cagr_statistics(f"Growth rate at ${arr}M (N={len(df_r)})", df_r)
    ex2 = st.beta_expander("Expand")
    format_dict = {'CAGR': '{0:.2f}%',
                   'Time (Yrs)': '{0:.2f}',
                   'Growth multiple': '{0:.2f}'}
    df_r = df_r[['ARR range', 'CAGR', 'Time (Yrs)', 'Growth multiple']].sort_values('Growth multiple', ascending=False)
    df_r = df_r.reset_index(1, drop=True)

    # df_r_style = df_r.style.applymap(highlight_gte_2,subset=['CAGR']).format(format_dict)
    def highlight_gte_2_row(val):
        color = 'yellow' if val['CAGR'] >= 100 else 'white'
        return [f"background-color: {color}"] * len(val)

    ex2.table(df_r.style.apply(highlight_gte_2_row, axis=1)
              .format(format_dict))
    # .background_gradient(cmap='Greens',subset=['CAGR','Growth multiple']))
    # ex2.table(df_r[df_r['CAGR'] > filter_val].sort_values('Growth multiple', ascending=False))
    ex2.table(df_r.style.format(format_dict))
    ex2.table(df_r)
    return


def highlight_gte_2_col(val):
    color = 'yellow' if type(val) == np.float and val >= 100 else 'white'
    return f'background-color: {color}'


def display_cohort_cagr_statistics(title, df, cagr_col='CAGR', time_col='Time (Yrs)'):
    def df_quantile(df, q, label):
        df_q = df.groupby('ARR range').agg({time_col: lambda x: x.quantile(1 - q),
                                            cagr_col: lambda x: x.quantile(q)})

        df_q['Quantile'] = label
        return df_q

    df_top_decile = df_quantile(df, 0.9, 'Top Decile')
    df_top_quartile = df_quantile(df, 0.75, 'Top Quartile')
    df_median = df_quantile(df, 0.5, 'Median')
    df_bot_quartile = df_quantile(df, 0.25, 'Bottom Quartile')
    df_top_decile.reset_index(inplace=True)
    df_median.reset_index(inplace=True)
    df_top_quartile.reset_index(inplace=True)
    df_bot_quartile.reset_index(inplace=True)

    x = pd.concat([df_bot_quartile, df_median, df_top_quartile, df_top_decile])
    x.set_index('Quantile', inplace=True)
    st.write(title)
    format_dict = {cagr_col: '{0:.2f}%', time_col: '{0:.2f}'}
    st.table(x.style.applymap(highlight_gte_2_col, subset=[cagr_col]).format(format_dict))
    return x


def create_quantile_ticker(ticker, current_cagr, current_arr, arr_hi, arr_lo, g_e=0.9):
    t = 0

    cagr = current_cagr
    a = np.array([t, cagr, current_arr]).reshape((1, 3))

    while current_arr > arr_hi:
        t = t + 1
        cagr = cagr / g_e
        current_arr = current_arr / (1 + cagr / 100)
        a = np.append(a, [[t, cagr, current_arr]], axis=0)

    if current_arr < arr_hi:
        t_x = fin.time_between_arr_ranges(end_arr=arr_hi, start_arr=current_arr, r=cagr) / 4.0
        a = np.append(a, [[t - t_x, cagr, arr_hi]], axis=0)

    while current_arr > arr_lo:
        t = t + 1
        cagr = cagr / g_e
        current_arr = current_arr / (1 + cagr / 100)
        a = np.append(a, [[t, cagr, current_arr]], axis=0)

    if current_arr < arr_lo:
        t_x = fin.time_between_arr_ranges(end_arr=arr_lo, start_arr=current_arr, r=cagr) / 4.0
        a = np.append(a, [[t - t_x, cagr, arr_lo]], axis=0)

    df = pd.DataFrame({'t': a[:, 0], 'CAGR': a[:, 1], 'ARR': a[:, 2]}, index=[ticker] * (t + 3)).sort_values('t',
                                                                                                             ascending=True)
    df.index.name = 'ticker'
    df['t'] = df['t'].max() - df['t']
    return df


def display_cohort_cagr_estimations(df_stats, cagr_col='CAGR', time_col='Time (Yrs)'):

    df_stats = df_stats.reset_index()
    median_cagr, top_quartile_cagr, top_decile_cagr = [df_stats.loc[df_stats['Quantile'] == i, 'CAGR'].iloc[0] for i in
                                                       ['Median', 'Top Quartile', 'Top Decile']]
    df_q = pd.concat([create_quantile_ticker('Top Decile', top_decile_cagr, 150, 100, 50, g_e=1),
                      create_quantile_ticker('Top Quartile', top_quartile_cagr, 150, 100, 50, g_e=1),
                      create_quantile_ticker('Median', median_cagr, 150, 100, 50, g_e=0.8)])

    df_qs = df_q.groupby('ticker') \
        .apply(lambda df_q: pd.DataFrame.from_dict(data={'ARR range': ['(50, 100]'],
                                                         'Time (Yrs)': [(df_q[df_q['ARR'] == 100]['t'][0] -
                                                                         df_q[df_q['ARR'] == 50]['t'][0])],
                                                         'CAGR': [fin.cagr(100, 50, (
                                                                 df_q[df_q['ARR'] == 100]['t'][0] -
                                                                 df_q[df_q['ARR'] == 50]['t'][0]) * 4)]})) \
        .reset_index(level=1, drop=True) \
        .reindex(['Median', 'Top Quartile', 'Top Decile'])
    format_dict = {cagr_col: '{0:.2f}%', time_col: '{0:.2f}'}
    st.write(f"**Estimated CAGR from $50M to $100M**")
    st.table(df_qs.style.applymap(highlight_gte_2_col, subset=[cagr_col]).format(format_dict))


def display_cohort_cagr(df_marker_index, df, arr_low, arr_hi, g_e=0):
    st.markdown("---")
    a0, a1 = (arr_low, arr_hi)
    interval_ranges = pd.IntervalIndex.from_tuples([(a0, a1)])
    df_r = cohort_cagr_over_marker_series(df=df,
                                          df_marker_index=df_marker_index, interval_ranges=interval_ranges)
    df_stats = display_cohort_cagr_statistics(
        f"**Time (yrs) to double ARR from ${arr_low}M to ${arr_hi}M  ** (N={len(df_r)})",
        df_r)
    if g_e:
        display_cohort_cagr_estimations(df_stats)
    ex2 = st.beta_expander("Expand")
    format_dict = {'CAGR': '{0:.2f}%',
                   'Time (Yrs)': '{0:.2f}',
                   'Row Time (Yrs)': '{0:.2f}',
                   'Growth multiple': '{0:.2f}'}
    df_r = df_r[['ARR range', 'CAGR', 'Time (Yrs)', 'Row Time (Yrs)', 'Growth multiple']] \
        .sort_values('Growth multiple', ascending=False) \
        .reset_index(1, drop=True)

    # df_r_style = df_r.style.applymap(highlight_gte_2,subset=['CAGR']).format(format_dict)
    def highlight_gte_2_row(val):
        color = 'yellow' if val['CAGR'] >= 100 else 'white'
        return [f"background-color: {color}"] * len(val)

    ex2.table(df_r[['CAGR', 'Time (Yrs)', 'Growth multiple']]
              .style.apply(highlight_gte_2_row, axis=1)
              .format(format_dict))

    # .to_excel(f'arr_{arr_low}_{arr_hi}.xlsx', engine='openpyxl'))
    # .background_gradient(cmap='Greens',subset=['CAGR','Growth multiple']))

    return df_r[['CAGR', 'Time (Yrs)', 'Growth multiple']]


def benchmark_cagr_over_ranges(df, df_marker_index):
    st.header('Range analysis')

    t_d = compute_cohort_cagr_panels(df_marker_index, df)
    st.markdown("***")

    # Display Growth rates over a range of ARR intervals
    display_cohort_cagr(df_marker_index, df, 40, 80)
    display_cohort_cagr(df_marker_index, df, 50, 100)
    display_cohort_cagr(df_marker_index, df, 150, 300, 0.9)
    display_cohort_cagr(df_marker_index, df, 200, 300)
    display_cohort_cagr(df_marker_index, df, 200, 400)
    a0 = st.number_input("ARR min", 50, 1000, 100)
    a1 = st.number_input("ARR max", 50, 1000, 200)
    display_cohort_cagr(df_marker_index, df, a0, a1)

    return


def benchmark_cagr_one_year_forward(df, df_marker_index):
    st.header('One year forward CAGR')

    st.markdown("***")
    #Display Growth rates over a range of ARR intervals
    display_cohort_cagr_one_year_forward(df_marker_index, df, 40)
    display_cohort_cagr_one_year_forward(df_marker_index, df, 50)
    display_cohort_cagr_one_year_forward(df_marker_index, df, 100)
    display_cohort_cagr_one_year_forward(df_marker_index, df, 150)
    display_cohort_cagr_one_year_forward(df_marker_index, df, 200)
    display_cohort_cagr_one_year_forward(df_marker_index, df, 300)
    display_cohort_cagr_one_year_forward(df_marker_index, df, 400)

    a0 = st.number_input("ARR min", 50, 1000, 100)
    display_cohort_cagr_one_year_forward(df_marker_index, df, a0)

    return


def list_private_cos():
    df = pd.read_csv('private_co_data.csv')
    df = df[df.columns[df.columns != 'Ticker']]
    df.set_index('Name', inplace=True)
    st.table(df)


def add_ipo_cohort_info(df, bins, labels):
    df_i = pd.read_csv('company_db.csv')
    df = df.join(other=df_i[['ticker', 'ipo_year', 'ipo_month']].set_index('ticker'), on='ticker').reset_index()
    df.loc[df['ticker'] == 'DB', 'ipo_year'] = 2022  # Fix me
    df.loc[df['ticker'] == 'DB', 'ipo_month'] = 0  # Fix me

    df['IPO date'] = df['ipo_year'] + df['ipo_month'] / 12
    df['IPO cohort'] = pd.cut(df['ipo_year'], bins, labels=labels)
    df['IPO cohort'] = df['IPO cohort'].astype('str')

    return df


axis_fmt_dict = dict(ARR="ARR ($M)", CAGR="CAGR %")

def custom_legend(fig, title, nameSwap, x=0.01,y=0.95,tx=0.5, ty=0.95):
    for i, dat in enumerate(fig.data):
        for elem in dat:
            if elem == 'name':
                fig.data[i].name = nameSwap[fig.data[i].name]

    fig = fig.update_layout(legend=dict(yanchor="top", y=y, xanchor="left", x=x))
    fig = fig.update_layout(title = {'text': title, 'y': ty, 'x': tx, 'xanchor': 'center', 'yanchor': 'top'})
    return (fig)


def ipo_legend_dict(df):
    # Add counts
    # Example: "2012-2016" --> "2012-2016 (15)"
    return {k: f"{k} ({v})" for (k, v) in df['IPO cohort'].value_counts().to_dict().items()}


def custom_annotation(fig, text, x, y, ax,ay):
    fig.add_annotation(
        x=x,
        y=y,
        xref="x",
        yref="y",
        text=text,
        showarrow=True,
        font=dict(
            family="Courier New, monospace",
            #size=16,
            #color="#ffffff"
        ),

        align="center",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        ax=7,
        ay=-50,
        arrowcolor="#636363",
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        #bgcolor="#ff7f0e",
        opacity=0.8
    )
    return fig

def plot_arr_over_time(df, df_r, arr0, arr1):

    def median_arr_across_ipo_cohorts(df_r,arr0,arr1):
        df_s = df_r.groupby('IPO cohort')['CAGR'].median().reset_index()
        num_quarters = int(np.max(np.ceil(fin.time_between_arr_ranges(arr1, arr0, df_s['CAGR'])) + 1))
        num_vintages = len(df_s)
        df_s['sort_key'] = np.arange(num_vintages)
        df_s = pd.concat([df_s] * num_quarters).sort_values('sort_key', ascending=True)
        df_s['t'] = list(range(num_quarters)) * num_vintages
        df_s['ARR'] = fin.fv(arr0, df_s['CAGR'], df_s['t'] / 4)
        df_s['ARR_p'] = df_s.groupby('IPO cohort')['ARR'].shift(1)
        df_s = df_s[(df_s['ARR'] <= arr1) | (df_s['ARR_p'] < arr1)]
        df_s['t'] = df_s['t']*3 # convert timeline from quarters to months
        return df_s

    df_s = median_arr_across_ipo_cohorts(df_r,arr0,arr1)
    df = df.reset_index()
    df = df[df['ticker'].isin(df_r['ticker'].unique())]
    df = df[(df['ARR'] >= arr0) & (df['ARR'] < arr1)]
    df['t2'] = df.groupby('ticker').transform('min')['t']
    df['t'] = df['t'] - df['t2']
    df = df[df['t'] < 5]
    df['t'] =df['t']*3 # convert timeline from quarters to months

    fig = px.line(df, x='t', y='ARR', color='ticker',labels = axis_fmt_dict)
    fig.update_traces({"line": {"color": "lightgrey"}})
    fig.update_layout(title=f"ARR Growth from ${arr0}M to ${arr1}M",
                      showlegend=False)
    fig.update_xaxes(range=[0, 12])
    dct = dict(zip(df_s['IPO cohort'].unique(), px.colors.qualitative.Plotly))
    tickers = list(df_r[df_r['IPO cohort'] == '2019-Today']['ticker'])

    dct = dict(zip(tickers, px.colors.qualitative.Plotly))
    for t in tickers:
        temp = df[df['ticker'] == t]
        fig.add_trace(go.Scatter(x=temp['t'], y=temp['ARR'],name=t)),
        # line=dict(color=dct[t], width=4)))

    st.plotly_chart(fig)

    return

def plots_at_ipo_time(bins, cohort_labels,df):
    df = add_ipo_cohort_info(df, bins=bins, labels=cohort_labels)
    df[['ARR_1yr_b', 't_1yr_b']] = df.groupby('ticker')[['ARR', 't']].shift(4)
    df['CAGR'] = fin.cagr(df['ARR'], df['ARR_1yr_b'], df['t'] - df['t_1yr_b']) #LTM CAGR

    df = df[df['t'] == 0] # Select time = IPO
    df = df.dropna(subset=['IPO cohort'])

    '''
    #####################################
    Figure 1
    Scatter plot: ARR against IPO date
    #####################################
    '''
    df_cagr = df.dropna(subset=['CAGR'])
    title = "ARR at IPO"
    fig = px.scatter(df_cagr,
                     x='IPO date',
                     y='ARR',
                     color='IPO cohort',
                     size='CAGR',
                     labels=axis_fmt_dict)
    fig = custom_legend(fig, title, ipo_legend_dict(df))
    # Add regression line
    df_reg = df[df['IPO cohort'] != '2019-Today']
    regline = sm.OLS(df_reg['ARR'], sm.add_constant(df_reg['IPO date'])).fit().fittedvalues
    fig.add_trace(go.Scatter(x=df_reg['IPO date'],
                             y=regline,
                             mode='lines',
                             marker_color='black',
                             name='Best-fit (2002-2018)',
                             line=dict(width=4, dash='dot')))
    # Add shaded circle
    fig.add_shape(type="circle",
                  xref="x", yref="y",
                  x0=2018.5, y0=200,
                  x1=2022, y1=700,
                  opacity=0.2,
                  fillcolor="LightSalmon",
                  line_color="LightSalmon",
                  )
    st.plotly_chart(fig)

    '''
    #################################################################
    Figure 2
    Scatter plot: Growth rate against IPO date with bubble size = ARR
    ################################################################
    '''
    title = "Growth rate (CAGR) at IPO"
    fig = px.scatter(df,
                     x='IPO date',
                     y='CAGR',
                     color='IPO cohort',
                     size='ARR',
                     labels=axis_fmt_dict)
    fig = custom_legend(fig, title, ipo_legend_dict(df))
    st.plotly_chart(fig)

    '''
    ################################################################
    Figure 3
    Scatter plot: LTM Growth rate against ARR at IPO
    ################################################################
    '''
    st.table(df['CAGR'].describe())
    title = "LTM Growth rate & ARR at IPO"
    fig = px.scatter(df,
                     x='ARR',
                     y='CAGR',
                     color='IPO cohort',
                     labels=axis_fmt_dict,
                     hover_name=df['ticker'])
    fig = custom_legend(fig, title, ipo_legend_dict(df))
    fig.add_shape(type="rect",
                  xref="x", yref="y",
                  x0=300, y0=df['CAGR'].quantile(0.58),
                  x1=600, y1=130,
                  opacity=0.2,
                  fillcolor="LightSalmon",
                  line_color="LightSalmon",
                  )
    st.plotly_chart(fig)

    '''
    ######################################################################
    Figure 4
    Scatter plot: LTM Growth rate against ARR at IPO for top quartile only
    ######################################################################
    '''
    df_topq = df[df['CAGR'] > df['CAGR'].quantile(0.75)]
    fig = px.scatter(df_topq,
                     x='ARR',
                     y='CAGR',
                     color='IPO cohort',
                     labels=axis_fmt_dict,
                     hover_name=df_topq['ticker'])
    fig = custom_legend(fig, title, ipo_legend_dict(df_topq))
    st.plotly_chart(fig)

    '''
    ######################################################################
    Figure 4
    Scatter plot: ARR at IPO (without bubbles for CAGR)
    ######################################################################
    '''
    title = 'ARR at IPO'
    fig = px.scatter(df,
                     x='IPO date',
                     y='ARR',
                     color='IPO cohort',
                     labels=axis_fmt_dict)
    fig = custom_legend(fig, title, ipo_legend_dict(df_topq))

    fig.add_shape(type="circle",
                  xref="x", yref="y",
                  x0=2018.5, y0=200,
                  x1=2022, y1=700,
                  opacity=0.2,
                  fillcolor="LightSalmon",
                  line_color="LightSalmon",
                  )

    df_reg = df[df['IPO cohort'] != '2019-Today']
    regline = sm.OLS(df_reg['ARR'], sm.add_constant(df_reg['IPO date'])).fit().fittedvalues
    fig.add_trace(go.Scatter(x=df_reg['IPO date'], y=regline,
                             name='Best-fit (2003-2015)',
                             mode='lines', marker_color='black', line=dict(width=4, dash='dot')))

    st.plotly_chart(fig)

    '''
    ######################################################################
    Figure 5
    Box plot: ARR at IPO by cohort
    ######################################################################
    '''
    fig = go.Figure()

    x = 0.25
    ax=20,
    ay=-30
    for l in cohort_labels:
        fig.add_trace(go.Box(y=df[df['IPO cohort'] == l]['ARR'],
                             boxpoints='all',
                             name=l))
        y = df[df['IPO cohort'] == l]['ARR'].median()
        fig = custom_annotation(fig, f"Median: ${y:.0f}M", x, y, ax, ay)
        x = x + 1

    fig.update_layout(yaxis_title='ARR ($M)',
                      showlegend=False)
    fig = custom_legend(fig,
                        "ARR at IPO by cohort",
                        ipo_legend_dict(df),
                        ty=0.85,tx=0.5)
    st.plotly_chart(fig)

    return


def plots_over_arr_ranges(bins, cohort_labels, df_marker_index, df):
    for a, b in [(200,400)]:
        df_r = display_cohort_cagr(df_marker_index, df, a, b, 0.9)
        df_r = add_ipo_cohort_info(df_r, bins=bins, labels=cohort_labels)
        df_r['Time (months)'] = df_r['Time (Yrs)'] * 12

        '''
        ######################################################################
        Figure 1
        Bar chart: Leaderboard: Fastest companies: Time taken from $a to $b
        ######################################################################
        '''
        fig = px.bar(df_r.head(10),
                     x="ticker",
                     y="Time (months)",
                     color="IPO cohort",
                     category_orders={"ticker": list(df_r['ticker'])},
                     orientation='v')
        fig = custom_legend(fig,
                            f"Fastest from ${a}M to ${b}M ARR",
                            ipo_legend_dict(df_r),
                            y=0.98,
                            tx=0.5,
                            ty=0.92)
        st.plotly_chart(fig)

        '''
        ######################################################################
        Figure 2
        Bar chart: Leaderboard: Fastest companies: CAGR taken from $a to $b
        ######################################################################
        '''
        fig = px.bar(df_r.head(10),
                     x="ticker",
                     y="CAGR",
                     color="IPO cohort",
                     category_orders={"ticker": list(df_r['ticker'])},
                     labels = axis_fmt_dict,
                     orientation='v')
        st.plotly_chart(fig)

        '''
        ######################################################################
        Figure 3
        Scatter plot: 
            y-axis: CAGR taken from $a to $b
            x-axis: IPO time
        ######################################################################
        '''
        title = f"Implied CAGR from ${a}M to ${b}M"
        fig = px.scatter(df_r,
                         x='IPO date',
                         y='CAGR',
                         color='IPO cohort',
                         labels = axis_fmt_dict)
        fig = custom_legend(fig, title, ipo_legend_dict(df_r))

        fig.add_shape(type="rect",
                      xref="x", yref="y",
                      x0=2018.5, y0=75,
                      x1=2022.3, y1=143,
                      opacity=0.2,
                      fillcolor="LightSalmon",
                      line_color="LightSalmon",
                      )
        st.plotly_chart(fig)

        '''
        ######################################################################
        Figure 4
        Histogram plot: CAGR taken from $a to $b
        ######################################################################
        '''
        fig = px.histogram(df_r,
                           x="CAGR",
                           color='IPO cohort',
                           barmode='overlay',
                           labels = axis_fmt_dict)
        st.plotly_chart(fig)

        '''
        ######################################################################
        Figure 5
        Box plot: CAGR taken from $a to $b by IPO Cohort
        ######################################################################
        '''
        fig = go.Figure()
        x = 0.2
        ax = 30,
        ay = -30
        for l in cohort_labels:
            fig.add_trace(go.Box(y=df_r[df_r['IPO cohort'] == l]['CAGR'],
                                 boxpoints='all',
                                 name=l))
            y = df_r[df_r['IPO cohort'] == l]['CAGR'].median()

            fig = custom_annotation(fig, f"Median: {y:.0f}%", x, y, ax, ay)
            x = x+1
        fig = custom_legend(fig, "Growth rate at IPO by cohort", ipo_legend_dict(df_r))
        fig.update_layout(
            title={
                'text':f'Implied CAGR from ${a}M to ${b}M ARR by IPO cohort',
                'y': 0.85,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            yaxis_title='CAGR %',
            showlegend=False)
        st.plotly_chart(fig)

    return

def benchmark_camera_ready(df):
    bins = [2002, 2015, 2018, 2022]
    cohort_labels = [f"{b0 + 1}-{b1}" for b0, b1 in zip(bins, bins[1:])]
    cohort_labels[-1] = f"{bins[-2] + 1}-Today"

    df = df[df['Status'] == 'Public']
    def preprocess(df):
        df[['ARR_p', 't_p']] = df.groupby('ticker')[['ARR', 't']].shift(1)
        df[['ARR_n', 't_n']] = df.groupby('ticker')[['ARR', 't']].shift(-1)
        df[['ARR_1yr_f', 't_1yr_f']] = df.groupby('ticker')[['ARR', 't']].shift(-4)

        df = df[['ticker', 't', 'ARR', 'ARR_p', 't_p', 'ARR_1yr_f', 't_1yr_f']]
        df = df.sort_values(['ticker', 't'], ascending=True)
        df['row_num'] = range(len(df))
        df_marker_index = create_metric_markers_index(df)
        df = df.set_index(['ticker', 'row_num'])
        return df, df_marker_index

    df_m = df
    df, df_marker_index = preprocess(df_m)

    choice = st.radio("Plots",('At IPO time','Over revenue ranges'))

    if choice == 'At IPO time':
        plots_at_ipo_time(bins, cohort_labels, df)
    else:
        plots_over_arr_ranges(bins, cohort_labels, df_marker_index, df)

    return


def benchmark_main(df, analysis):
    if analysis == 'Camera-ready':
        return benchmark_camera_ready(df)

    list_private_cos()
    df.loc[df['ticker'] == 'DB', 'ticker'] = 'Private'
    selected_status = st.sidebar.selectbox('Type', ('All', 'Public', 'Private'))
    if selected_status != 'All':
        df = df[df['Status'] == selected_status]
        if st.sidebar.checkbox('Exclude recent IPOs'):
            df = df[df['Recent IPO'] == 'No']

    st.sidebar.info(f"Showing {selected_status} companies")

    df[['ARR_p', 't_p']] = df.groupby('ticker')[['ARR', 't']].shift(1)
    df[['ARR_n', 't_n']] = df.groupby('ticker')[['ARR', 't']].shift(-1)
    df[['ARR_1yr_f', 't_1yr_f']] = df.groupby('ticker')[['ARR', 't']].shift(-4)

    df = df[['ticker', 't', 'ARR', 'ARR_p', 't_p', 'ARR_1yr_f', 't_1yr_f']]
    df = df.sort_values(['ticker', 't'], ascending=True)
    df['row_num'] = range(len(df))
    df_marker_index = create_metric_markers_index(df)

    all_tickers = df['ticker'].unique()
    st.write(f"Total companies = {df['ticker'].nunique()}")
    default_tickers = all_tickers
    df = df.set_index(['ticker', 'row_num'])
    if analysis == 'ARR range':
        benchmark_cagr_over_ranges(df=df, df_marker_index=df_marker_index)
    elif analysis == '1yr CAGR at ARR':
        benchmark_cagr_one_year_forward(df=df, df_marker_index=df_marker_index)
    else:
        st.write("Range analysis")

    return
