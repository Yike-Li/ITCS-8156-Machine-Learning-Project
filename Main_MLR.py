"""
This main file includes data preprocessing and modeling
"""
# import
import numpy as np
import pandas as pd
import sys, os, patsy
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import seaborn as sns
from math import pi, floor, ceil
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
# plt.rcParams["font.family"] = "Times New Roman"

# set pandas printing
pd.set_option('display.max_rows', 1500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
pd.options.display.max_colwidth = 100

# set path
user = 'Administrator'
sys.path.insert(1, f'C:/Users/{user}/PycharmProjects/MyTest/CS ML/Term Project')  # from main folder
os.chdir(f'C:/Users/{user}/PycharmProjects/MyTest/CS ML/Term Project')
# noinspection PyUnresolvedReferences
from Model_Fit_Func import MAPE, add_trend, wss, add_sync_sin_days, benchmark_perf, shift_solar_terms

# %% import & preprocess
wf = pd.read_hdf(f'load_and_weather.h5', 'load_wbfcst')  # load w/ weather fcst, 2013-2018
wa = pd.read_hdf(f'load_and_weather.h5', 'fulldata')  # load w/ weather actual, 2004-2018

droplist = [f'T{i}' for i in range(21, 28 + 1)] + ['T12', 'T14'] + [f'RH{i}' for i in range(21, 28 + 1)] + \
           ['RH12', 'RH14'] + [f'WS{i}' for i in range(21, 28 + 1)] + ['WS12', 'WS14']
wa.drop(droplist, axis=1, inplace=True)  # drop stations not in wf
wa.columns = wa.columns[:5].tolist() + [f'T{i}' for i in range(1, 18 + 1)] + [f'RH{i}' for i in range(1, 18 + 1)] + \
             [f'WS{i}' for i in range(1, 18 + 1)]  # rename T, RH, WS cols
yrs = np.arange(2010, 2018 + 1, 1)  # 2010-2018. first 3 yrs for training

# %% Weather station selection
# wss_list = []
# for i in range(6):  # 6
#     subset_yr = yrs[i:4 + i]
#     df = wa[wa.year.isin(subset_yr)].copy()  # 选4年
#     df = add_trend(df)
#     df['weekday'] = df.date.dt.weekday
#     # store WSS in txt, set use_wss_list=False, 需提前清空wss_list=[]
#     df = wss(df, use_wss_list=False, tot_sta_num=18, idx=i)
#     with open("wss_list_18.txt", "w") as file:
#         for s in wss_list:
#             file.write(str(s) + "\n")

wss_list = []  # set wss_list to empty list
with open("wss_list.txt", "r") as file:  # use the current wss from txt file
    for line in file:
        wss_list.append(eval(line.strip()))  # eval转str为list

# %% Ex-post / Ex-ante Experiments
fcst_type = 'ex-ante'
columns = ['year', 'type', 'method', 'mape', 'SD', 'params', 'shift_deg']
result_df = pd.DataFrame(columns=columns)  # create empty dataframe to store results

for i in range(1):  # 6, 2013-2018
    subset_yr = yrs[i:4 + i]
    if fcst_type == 'ex-ante':
        dfa = wa[wa.year.isin(subset_yr[:-1])].copy()  # 选4年, df_actual
        dff = wf[wf.year == subset_yr[-1]].copy()  # 选4年, df_actual
        df = pd.concat([dfa, dff], sort=True)

    else:  # if ex post, use all data from w_actual
        df = wa[wa.year.isin(subset_yr)].copy()

    df['dayofyr'] = df.date.dt.dayofyear
    df['daysinyr'] = np.where(df.date.dt.is_leap_year, 366, 365)
    df['month'] = df.date.dt.month
    df['weekday'] = df.date.dt.dayofweek
    df['year'] = df.date.dt.year
    df['hour'] = df.date.dt.hour
    # df['hrofyr'] = (df.dayofyr - 1) * 24 + df.hour + 1  # get hour of year
    # df['hrsinyr'] = np.where(df.date.dt.is_leap_year, 366 * 24, 365 * 24)
    df = add_trend(df)
    df = wss(df, process_weather=True, weather_var=('RH', 'WS'), wss_list=wss_list, idx=i)
    df['RHS'] = np.where(df.month.isin([6, 7, 8, 9]), df['RH'], 0)  # summer of RH paper model, 6,7,8,9
    df['WSS'] = np.where(df.month.isin([6, 7, 8]), df['WS'] ** 0.16, 0)  # summer of WS paper model, 6,7,8

    for fit_test in [3]:  # always use 3 yrs to train
        print(f'\nTest year={subset_yr[-1]}, training year={subset_yr[:-1]}, fit_test={fit_test}')

        # Recency
        for h in [1]:  # [0, 1, 2]
            for d in [1]:  # [0, 1, 2]
                ## benchmark (month) models
                # benchmark_perf(df, fit_test, bm_model='Vanilla', hmax=h, dmax=d)
                # benchmark_perf(df, fit_test, bm_model='RH', hmax=h, dmax=d)
                # benchmark_perf(df, fit_test, bm_model='WS', hmax=h, dmax=d)
                benchmark_perf(df, fit_test, bm_model=['RH', 'WS'], hmax=h, dmax=d)

                # test set上遍历所有度数
                for j in range(1):  # j=0, no shifting start dates
                    ve_day_dict = shift_solar_terms(shiftDeg=15 * j)
                    # 1 sine
                    # add_sync_sin_days(df, hmax=h, dmax=d, sin2t=False, sin05t=False)
                    # 2 sines
                    # add_sync_sin_days(df, hmax=h, dmax=d, sin05t=False)
                    # 3 sines
                    # add_sync_sin_days(df, hmax=h, dmax=d)
                    #
                    # add_sync_sin_days(df, RH1t=True, hmax=h, dmax=d)
                    # add_sync_sin_days(df, WS1t=True, hmax=h, dmax=d)
                    add_sync_sin_days(df, RH1t=True, WS1t=True, hmax=h, dmax=d)

        # Data-driven
        # add_sync_sin_days(df, fit_test, use_eq=False, )
        # add_sync_sin_days(df, fit_test, use_eq=False, RH1t=True, )
        # add_sync_sin_days(df, fit_test, use_eq=False, WS1t=True, )
        # add_sync_sin_days(df, fit_test, use_eq=False, RH1t=True, WS1t=True, )

# %% post-processing
result_df.drop_duplicates(subset=['year', 'type', 'method', 'shift_deg'], inplace=True)
result_test = result_df[result_df.type != 0].copy()
result_test['params'] = result_test['params'].astype(str).astype(int)  # conv to int
# result_test['SD'] = result_test['SD'].astype(str).astype(int)  # conv to int
result_test['year'] = result_test['year'].astype(str).astype(int)  # conv to int
result_test['shift_deg'] = result_test['shift_deg'].astype(str).astype(int)  # conv to int

summary = result_test.groupby(['method', 'shift_deg'], as_index=False).mean()  # 求测试集平均MAPE
print(summary)
# 求所有starting position中mape最小的
summary['name'] = summary['method'].str.split('|').str[0]  # unpack, take the 1st ele, add a name col
summary.groupby(['name'], as_index=False).min()

# print(summary)

