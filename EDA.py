"""
Gives EDA of the load and weather data.
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

wss_list = []  # set wss_list to empty list
with open(f"wss_list.txt", "r") as file:
    for line in file:
        wss_list.append(eval(line.strip()))  # eval转str为list
# %% load & temperature time series, after wss
for i in range(6):
    wss_wf = wss(wf[wf.year == i + 2013], process_weather=True, weather_var=('RH', 'WS'), wss_list=wss_list, idx=i)
    wss_wa = wss(wa[wa.year == i + 2013], process_weather=True, weather_var=('RH', 'WS'), wss_list=wss_list, idx=i)
    if i == 0:
        weather_comp = wss_wa.merge(wss_wf[['date', 'T', 'RH', 'WS']], how='left', on='date', suffixes=('_a', '_f'))
    else:
        weather_comp = pd.concat(
            [weather_comp,
             wss_wa.merge(wss_wf[['date', 'T', 'RH', 'WS']], how='left', on='date', suffixes=('_a', '_f'))],
            axis=0)

fig = plt.figure(figsize=(10, 6), dpi=150)
plt.subplot(211)  # Hourly load plot
plt.plot(weather_comp.date, weather_comp.load, linewidth=0.5, c='C0')
plt.ylabel('Load (kW)')
plt.title(f'Load and Actual Temperature Time Series')

plt.subplot(212)  # Hourly temperature plot
plt.plot(weather_comp.date, weather_comp.T_a, linewidth=0.5, c='C1')
plt.ylabel('Temperature (°F)')
plt.xlabel('Date')
# plt.savefig(f'Load & Temperature.png', bbox_inches='tight')
plt.show()

# %% Load vs. Temp
plt.figure(figsize=(8, 5), dpi=100)
plt.scatter(weather_comp.T_a, weather_comp.load, s=0.5, marker="o", c='royalblue')
# plt.axvline(x=60, color='red', linestyle=':', linewidth=3)
# plt.title('GEFCom2012 Zone 21 Load vs. Temperature')
# plt.xlim(3, 102)
# plt.ylim(.7, 3.5)
plt.xlabel('Temperature (°F)')
plt.ylabel('Load (kW)')
plt.show()
# %% Temp fcst error time series
weather_comp['T_ae'] = abs(weather_comp.T_a - weather_comp.T_f)
# plt.close('all')
fig = plt.figure(figsize=(10, 4), dpi=150)
plt.scatter(weather_comp.date, weather_comp.T_ae, s=1, c='C4')
# plt.xlabel('Date')
plt.ylabel('Absolute Error (°F)')
for i in range(6):
    plt.axvline(x=datetime(2013 + i, 1, 1), c='gray', linestyle=':', linewidth=0.5)
plt.axhline(y=10, c='black', linestyle='--', linewidth=1)
plt.title(f'Temperature Forecast Absolute Error')
plt.xlabel('Date')
# plt.savefig(f'graphs/Temp AE.png', bbox_inches='tight')
plt.show()

# %% EDA of RH and WS
# ds = full[full.year.isin(yrs[-4:])].copy()  # create data sample
# ds = wss(ds, process_weather=True, weather_var=('RH', 'WS'))  # conduct wss
# ds['WS016'] = ds['WS'] ** 0.16
# for year in yrs[-4:]:
#     ds.loc[ds.year == year, 'new_dayofyr'] = np.roll(ds.loc[ds.year == year, 'dayofyr'],
#                                                      (ve_day_dict[year] - 1) * 24)
# ds.loc[:, 'sin1t'] = np.sin((1 * 2 * pi / ds.daysinyr) * ds.new_dayofyr)
# ds['RHsin1t'] = ds['RH'] * ds['sin1t']
# ds['WSsin1t'] = ds['WS'] * ds['sin1t']
# weather_comp = ds[ds.year == np.unique(ds.year)[-1]]  # 只取一年来画图
df_plt = weather_comp[weather_comp.year == np.unique(weather_comp.year)[0]].copy()  # take 2013 only
df_plt['WS016'] = df_plt['WS_a']**0.16

def plot_weather_by_mon(x='RH_a', y='load', x_lab='Relative Humidity (%)', y_lab='Load (MW)', title=None, c='C0'):
    global df_plt, yrs
    fig, ax = plt.subplots(nrows=3, ncols=4, sharex='col', sharey='row', figsize=(9, 7), dpi=200)
    # fig.subplots_adjust(top=0.8)
    ax1 = fig.add_subplot(111, frameon=False)  # add a big axes, hide frame
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False,
                    right=False)  # hide tick and tick labels
    plt.xlabel(x_lab, fontsize=13)
    plt.ylabel(y_lab, fontsize=13)
    ax1.get_yaxis().set_label_coords(-0.06, 0.5)  # adjust y label location of big axes
    if title is not None:
        fig.suptitle(f'{title}', fontsize=12)
    for i in range(4):  # col
        for j in range(3):  # row
            ax[j, i].scatter(df_plt.loc[df_plt['month'] == i + j * 4 + 1][x],
                             df_plt.loc[df_plt['month'] == i + j * 4 + 1][y], s=2, c=c, alpha=0.7)
            ax[j, i].set_title('Month=' + str(i + j * 4 + 1), fontsize=11)
            # ax[j, i].set(xlim=(-10, 110), ylim=(0.5, 3.5))  # set xlim and ylim for each subplot
            ax[j, i].grid(True)  # add gridlines

            if j != 2:
                ax[j, i].xaxis.set_ticks_position('none')  # remove xsticks on first 2 rows, keep the last row ticks

    plt.savefig('plot_save.png', bbox_inches='tight')  # save plot, 'tight' fits for subplots
    # plt.show()


# RH
plot_weather_by_mon(x='RH_a', y='load', x_lab='Relative Humidity (%)', y_lab='Load (kW)',
                    c='C3')  # 'Load vs. Actual RH by Month'
# plot_weather_by_mon(x='RHsin1t', y='load', x_lab='RH * sin1t', y_lab='Load (MW)', title='Load vs. RHsin1t by Month')
plot_weather_by_mon(x='RH_a', y='T_a', x_lab='Relative Humidity (%)', y_lab='Temperature (°F)')  # Temp vs. RH by month
plot_weather_by_mon(x='hour', y='RH_a', x_lab='Hour', y_lab='Relative Humidity (%)')  # RH vs. hour by month

# RHsin1t / WSsin1t plot
var = 'WSsin1t'
fig, ax = plt.subplots(figsize=(10, 4), dpi=200)
ax.plot(df_plt['date'], df_plt[var].clip(lower=0), lw=1, c='C1')
plt.xlabel('Date', fontsize=12)
plt.ylabel(var + ' neg->0', fontsize=12)
plt.title(f'Time series plot of {var}, neg->0', fontsize=12)
plt.axhline(0, c='black', lw=1, alpha=0.5, linestyle=':')
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))  # set ticks every 2 months
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))  # set major ticks format
plt.savefig('plot_save.png', bbox_inches='tight')

# WS
plot_weather_by_mon(x='WS_a', y='load', x_lab='Wind Speed (mile/hr)', y_lab='Load (kW)',
                    c='C2')  # 'Load vs. WS by Month'
plot_weather_by_mon(x='WS016', y='load', x_lab='Wind Speed (mile/hr)', y_lab='Load (kW)',
                    c='C2')  # 'Load vs. WS**0.16 by Month'
plot_weather_by_mon(x='WS', y='T', x_lab='Wind Speed (mile/hr)', y_lab='Temperature (°F)')  # Load vs. WS by month
plot_weather_by_mon(x='hour', y='WS016', x_lab='Hour', y_lab='Wind Speed (mile/hr)')  # WS vs. hour by month

# %% load & temperature moving average
ma_window = 7
win_type = None  # None or 'triang'
i = 5
subset_yr = yrs[i:4 + i]
df = wa[wa.year.isin(subset_yr)].copy()
df['dayofyr'] = df.date.dt.dayofyear
df['daysinyr'] = np.where(df.date.dt.is_leap_year, 366, 365)
df['month'] = df.date.dt.month
df['weekday'] = df.date.dt.dayofweek
df['year'] = df.date.dt.year
df['hour'] = df.date.dt.hour
df['hrofyr'] = (df.dayofyr - 1) * 24 + df.hour + 1  # get hour of year
df['hrsinyr'] = np.where(df.date.dt.is_leap_year, 366 * 24, 365 * 24)
df = add_trend(df)
# df.dropna(how='any', inplace=True)  # before wss, drop rows with any nan。不用drop了因为没有nan了
df = wss(df, process_weather=True, weather_var=('RH', 'WS'), use_wss_list=True, tot_sta_num=18, idx=i)
# df['RHS'] = np.where(df.month.isin([6, 7, 8, 9]), df['RH'], 0)  # summer of RH paper model, 6,7,8,9
# df['WSS'] = np.where(df.month.isin([6, 7, 8]), df['WS'] ** 0.16, 0)  # summer of WS paper model, 6,7,8


sine_adjust = 50  # 10000 for ISONE
sine_factor = 40  # 1500 for ISONE
shift_Deg = 0  # deg shifted from VE
shift_day = 79 - 1 + round(shift_Deg / 360 * 365)

for year in np.unique(df.year):
    df.loc[df.year == year, 'new_dayofyr'] = np.roll(df.loc[df.year == year, 'dayofyr'], shift_day * 24)
df.loc[:, 'sin1t'] = np.sin((1 * 2 * pi / df.daysinyr) * df.new_dayofyr) * sine_factor + sine_adjust
for year in np.unique(df.year):
    df.loc[df.year == year, 'new_dayofyr'] = np.roll(df.loc[df.year == year, 'dayofyr'], shift_day * 24)
df.loc[:, 'sin2t'] = -(np.sin((2 * 2 * pi / df.daysinyr) * df.new_dayofyr) * sine_factor + sine_adjust)
for year in np.unique(df.year):
    df.loc[df.year == year, 'new_dayofyr'] = np.roll(df.loc[df.year == year, 'dayofyr'], shift_day * 24)
df.loc[:, 'sin05t'] = np.sin((0.5 * 2 * pi / df.daysinyr) * df.new_dayofyr) * sine_factor + sine_adjust

df.loc[:, 'sin05t-sin1t'] = (df['sin05t'] - df['sin1t']) + sine_adjust  # summer peak ISONE
df.loc[:, 'sin05t+sin1t'] = (df['sin05t'] + df['sin1t']) - sine_adjust  # winter peak VT
df.loc[:, 'sin05t+sin2t'] = (df['sin05t'] + df['sin2t']) - sine_adjust
df.loc[:, 'sin1t*T'] = (df['sin1t'] - sine_adjust) * df['T'] / sine_factor

df['load_ma'] = df.load.rolling(24 * ma_window, center=True, win_type=win_type).mean() / 2
df['T_ma'] = 2 * df['T'].rolling(24 * ma_window, center=True, win_type=win_type).mean() + 40
df = df[(df.date >= datetime(2016, 3, 20)) & (df.date <= datetime(2017, 3, 20))]  # subset df for a year

fig, ax = plt.subplots(figsize=(10, 2.5), dpi=100)
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.yaxis.set_ticks([])  # remove y ticks
ax.xaxis.set_ticks([])  # remove y ticks
# ax.xaxis.set_ticks_position('bottom')  # Only show ticks on the bottom spines

plt.plot(df.date, df.load_ma, linewidth=2, label='Load')
plt.plot(df.date, df.T_ma, label='Temperature', c='C1')
# plt.plot(df.date, df['T'].rolling(24 * ma_window, center=True, win_type=win_type).mean(), label=f'{region}')

# plt.plot(df.date, df['sin1t'], c='purple', linewidth=2, label=f'sin1t ({shift_Deg}°)')
# plt.plot(df.date, df['sin1t*T'], c='black', linewidth=0.5, label=f'sin1t * T ({shift_Deg}°)')
# plt.plot(df.date, df['sin2t'], c='blue', linewidth=0.2, label=f'sin2t ({shift_Deg}°)')
# plt.plot(df.date, df['sin05t'], c='brown', linewidth=0.2, label=f'sin05t ({shift_Deg}°)')
# plt.plot(df.date, df['sin05t+sin2t'], c='purple', linewidth=2, label=f'sin05t+sin2t ({shift_Deg}°)')
# plt.plot(df.date, df['sin05t-sin1t'], c='purple', linewidth=2, label=f'sin05t-sin1t ({shift_Deg}°)')# winter peak VT
# plt.plot(df.date, df['sin05t-sin1t'], c='purple', linewidth=2, label=f'sin05t+sin1t ({shift_Deg}°)')  # summer pk ISONE

# tmp = df['T'].rolling(24 * 30, center=True, win_type='triang').mean()
# (tmp.iloc[24 * (79 + 61) + 365 * 0 * 24] + tmp.iloc[24 * (79 + 61) + 365 * 1 * 24] + tmp.iloc[
#     24 * (79 + 61) + 365 * 2 * 24]) / 3
# (tmp.iloc[24 * (79 + 259) + 365 * 0 * 24] + tmp.iloc[24 * (79 + 259) + 365 * 1 * 24] + tmp.iloc[
#     24 * (79 + 259) + 365 * 2 * 24]) / 3

# for i in subset_yr:
#     plt.axvline(x=datetime(i, 1, 1), c='gray', linestyle=':', linewidth=.5)
plt.axvline(x=datetime(2016, 3, 20), c='C0', linestyle='--', linewidth=.7, label='VE (0°)', alpha=0.1)
# plt.axvline(x=datetime(i, 3, 20) + timedelta(days=61), c='green', linestyle='--', linewidth=1, label='60°')
plt.axvline(x=datetime(2016, 6, 21), c='C0', linestyle='--', linewidth=.7, label='Ssol (90°)', alpha=0.1)
plt.axvline(x=datetime(2016, 9, 23), c='C0', linestyle='--', linewidth=.7, label='AE (180°)', alpha=0.1)
plt.axvline(x=datetime(2016, 12, 21), c='C0', linestyle='--', linewidth=.7, label='Wsol (270°)', alpha=0.1)
plt.axvline(x=datetime(2017, 3, 20), c='C0', linestyle='--', linewidth=.7, label='VE (0°)', alpha=0.1)
# plt.axvline(x=datetime(i, 12, 21 - 15), c='magenta', linestyle='--', linewidth=1, label='255°')

# plt.axhline(y=60, c='green', linestyle='-', linewidth=1)
# plt.axhline(y=36, c='magenta', linestyle='-', linewidth=1, label='36°F')
# plt.ylim(20, 180)
# plt.ylim(40, 180)
# plt.title(f'{region} (window={ma_window} days) of Load and Temperature')
# remove dup legend for VE and
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
# plt.legend(by_label.values(), by_label.keys(), loc='upper right')
plt.show()

# %% plot predict load curves, Month vs. Sine

i = 4 # plot第i+1个test year
j = 0
plot_date = '2017-01-28'
fcst_type = 'ex-ante'
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
df = add_trend(df)
df = wss(df, process_weather=True, weather_var=('RH', 'WS'), wss_list=wss_list, idx=i)
df['RHS'] = np.where(df.month.isin([6, 7, 8, 9]), df['RH'], 0)  # summer of RH paper model, 6,7,8,9
df['WSS'] = np.where(df.month.isin([6, 7, 8]), df['WS'] ** 0.16, 0)  # summer of WS paper model, 6,7,8

plt.figure(figsize=(12, 5))

h, d = 1, 1
benchmark_perf(df, fit_test=3, bm_model='Vanilla', hmax=h, dmax=d, plot_pred=True, plot_date=plot_date)
h, d = 1, 2
add_sync_sin_days(df, hmax=h, dmax=d, plot_pred=True, plot_date=plot_date)
ve_day_dict = shift_solar_terms(shiftDeg=15 * j)
plt.show()
