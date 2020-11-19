"""
Functions used in the main file.

"""

import numpy as np
import pandas as pd
from datetime import date, timedelta, datetime
import sys, os, patsy, time
import statsmodels.api as sm
import matplotlib.pyplot as plt
from math import pi, floor, ceil

# import equinox and solstice info
equinox = pd.read_excel(f'Project input.xlsx', sheet_name='Equi & Sols in EST')
equinox['hrofyr'] = (equinox.date.dt.dayofyear - 1) * 24 + equinox.hour + 1
equinox['dayofyr'] = equinox.date.dt.dayofyear
ve, ssol, ae, wsol = equinox[equinox.type == 'VE'], equinox[equinox.type == 'Ssol'], equinox[equinox.type == 'AE'], \
                     equinox[equinox.type == 'Wsol']

ve_day_dict = dict(zip(ve.year, ve.dayofyr))
ss_day_dict = dict(zip(ssol.year, ssol.dayofyr))
ae_day_dict = dict(zip(ae.year, ae.dayofyr))
ws_day_dict = dict(zip(wsol.year, wsol.dayofyr))


def add_trend(df, log=0):
    """Add Linear trend, log_trend col to dataset. """
    df = df.reset_index(drop=True)  # reset index before concatenate
    trend = pd.Series(np.arange(1, len(df) + 1, 1, dtype=int), name='trend')
    if log == 1:
        logtrend = np.log(pd.Series(np.arange(1, len(df) + 1, 1, dtype=int), name='logtrend'))
    return pd.concat([df, trend, logtrend], axis=1) if log == 1 else pd.concat([df, trend], axis=1)


def MAPE(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true).flatten(), np.asarray(y_pred).flatten()
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def select_day(name, w, start, end, tot_steps, yr_list, data, f, fit_test, plot_pred=False):
    """
    Abandoned.
    Parameters
    ----------
    name: sin1t, sin2t, etc.
    w: factor to adjust sine wave frequency
    start
    end
    tot_steps
    yr_list
    data: input data
    f: formula
    fit_test: years to use for parameter estimation
    plot_pred: if need to plot prediction

    Returns: data with updated sinXt col
    -------

    """
    mape_list = []  # store mape of each shift day
    # select shift day and create sin col
    for i in range(start, end, ceil((end - start) / tot_steps)):

        for year in yr_list:
            data.loc[data.year == year, 'new_dayofyr'] = np.roll(data.loc[data.year == year, 'dayofyr'], i * 24)

        data.loc[:, name] = np.sin((w * 2 * pi / data.daysinyr) * data.new_dayofyr)
        train = data[data.year != yr_list[-1]].copy()

        mape = []

        for year in np.unique(train.year):  # Conduct CV
            tr = train[train.year != year]  # use previous 2 years as training set
            val = train[train.year == year]  # val set
            y_tr, X_tr = patsy.dmatrices(f, tr, return_type='dataframe')  # training
            y_val, X_val = patsy.dmatrices(f, val, return_type='dataframe')  # validation, is on training + val set
            model = sm.OLS(y_tr, X_tr).fit()  # use training period to fit model
            mape.append(MAPE(y_val, model.predict(X_val)))
        mape_list.append((i, np.mean(mape)))  # store (shift_day, mean_cv_mape)
        print(f'{name} shift day={i}, Average MAPE across {len(yr_list) - 1}-fold CV: {np.mean(mape):.3f}')

    mape_list.sort(key=lambda tup: tup[1])  # sort ascending by the second column: cv mape
    best_day, best_cv_mape = mape_list[0][0], mape_list[0][1]
    print(f'Chosen {name} shift day={best_day}, {len(yr_list) - 1}-fold CV MAPE={best_cv_mape:.3f}')

    # update data with selected shift day
    for year in yr_list:
        data.loc[data.year == year, 'new_dayofyr'] = np.roll(data.loc[data.year == year, 'dayofyr'], best_day * 24)

    data.loc[:, name] = np.sin((w * 2 * pi / data.daysinyr) * data.new_dayofyr)

    print(f'Test set performance use recent {fit_test} years for param estimation.')
    y_tr, X_tr = patsy.dmatrices(f, data[data.year.isin(yr_list[-fit_test - 1:-1])], return_type='dataframe')
    y_test, X_test = patsy.dmatrices(f, data[data.year == yr_list[-1]], return_type='dataframe')
    model = sm.OLS(y_tr, X_tr).fit()  # use whole training period to fit model
    mape_test = MAPE(y_test, model.predict(X_test))
    print(f'{name} shift day={best_day}, Test MAPE: {mape_test:.3f}')
    print(f'---------------------------------------------------------------------------')

    if plot_pred:
        plt.figure(figsize=(12, 5))
        plt.plot(model.predict(X_test), label='predict')
        plt.plot(y_test, label='actual')
        plt.legend()
        plt.show()

    return data


def benchmark_perf(data: pd.DataFrame, fit_test, bm_model='Vanilla', hmax=0, dmax=0, show_CV=False,
                   plot_pred=False, plot_date='2013-03-26'):
    """
    bm_model: 'Vanilla', 'RH', 'WS' or ('RH','WS').
    Return: CV_mape and test_mape for benchmark model
    """
    global result_df, subset_yr
    yr_list = np.unique(data.year)
    T_term = '+ C(month) * T + C(month) * I(T**2) + C(month) * I(T**3) + C(hour) * T + C(hour) * I(T**2) + C(hour) * I(T**3) '
    f = 'load ~ trend + C(hour) * C(weekday) + C(month) ' + T_term
    method = 'Vanilla '

    if 'RH' in bm_model:
        RH_term = '+ RHS * T + RHS * I(T**2) + I(RHS**2) * T + I(RHS**2) * I(T**2) + C(hour) * RHS + C(hour) * I(RHS**2) '
        f += RH_term
        method += '+ RH '
    if 'WS' in bm_model:
        WS_term = '+ WSS * T + WSS * C(hour) '
        f += WS_term
        method += '+ WS'

    # print(f)
    if hmax + dmax > 0:  # if either hmax or dmax >0, execute below
        data, f = recency(data, f, hmax, dmax)
    # print(f)

    if show_CV:
        train = data[data.year != yr_list[-1]].copy()
        mape = []
        for year in np.unique(train.year):  # Conduct CV
            tr = train[train.year != year]  # use previous 2 years as training set
            val = train[train.year == year]  # val set
            y_tr, X_tr = patsy.dmatrices(f, tr, return_type='dataframe')  # training
            y_val, X_val = patsy.dmatrices(f, val, return_type='dataframe')  # validation, is on training + val set
            model = sm.OLS(y_tr, X_tr).fit()  # use training period to fit model
            mape.append(MAPE(y_val, model.predict(X_val)))
        print(f'{bm_model} model (w/ month), Average MAPE across {len(yr_list) - 1}-fold CV: {np.mean(mape):.3f}')

        result_df = result_df.append(
            {'year': subset_yr[-1], 'type': 0, 'method': method + f'|h{hmax}d{dmax}', 'mape': np.mean(mape),
             'params': len(model.params), 'shift_deg': 999}, ignore_index=True)

    y_tr, X_tr = patsy.dmatrices(f, data[data.year.isin(yr_list[-fit_test - 1:-1])], return_type='dataframe')
    y_test, X_test = patsy.dmatrices(f, data[data.year == yr_list[-1]], return_type='dataframe')

    # time_start = time.time() # examine the training time
    model = sm.OLS(y_tr, X_tr).fit()  # use whole training period to fit model
    # print(f'Training time={time.time() - time_start:.2f} seconds.')

    print(f'No. of non-zero parameters: {len(model.params)}')
    mape_test = MAPE(y_test, model.predict(X_test))
    print(f'{bm_model} model (w/ month), Test MAPE: {mape_test:.3f}')
    print(f'---------------------------------------------------------------------------')

    result_df = result_df.append(
        {'year': subset_yr[-1], 'type': fit_test, 'method': method + f'|h{hmax}d{dmax}', 'mape': mape_test,
         'params': len(model.params), 'shift_deg': 999}, ignore_index=True)

    if plot_pred:
        test = data[data.year == yr_list[-1]]
        base = datetime(test.year.unique()[0], 1, 1)
        date = np.array([base + timedelta(hours=i) for i in range(1, len(test) + 1)])
        data_plot = pd.concat([y_test, model.predict(X_test)], axis=1).reset_index(drop=True)
        data_plot.rename(columns={'load': 'Actual', 0: 'Predict'}, inplace=True)
        data_plot = data_plot.set_index(date)

        # extract yr, month, day
        plot_date = list(map(int, plot_date.split('-')))
        date_from = datetime(plot_date[0], plot_date[1], plot_date[2])
        date_to = date_from + timedelta(days=7)

        plt.plot(data_plot.Actual.loc[date_from:date_to], color='black', label='Actual')  # plot actual, use index
        plt.plot(data_plot.Predict[date_from:date_to], marker='^', linestyle=':', label=f'Month model prediction')
        # remove dup legend, 2 Actuals created by 2 loops
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='best')

        # plt.title(f'Performance on a week starting {plot_date}')

        plt.xlabel('Date')
        plt.ylabel('Load (kW)')


def recency(data: pd.DataFrame, f, hmax, dmax):
    """
    Use this function after the formula f is finalized.
    f: input formula.
    Return: updated data and formula f.
    """
    base = 'load ~ trend + C(hour) * C(weekday) '  # 最基本model, 不含month不含T
    var = f[len(base):]  # store partial f in var, length=36, anything after base
    base += var

    # create lagged hourly temperature
    if hmax > 0:
        # create lagged hourly temperature
        for i in range(1, hmax + 1):
            colname = 'Th_' + str(i)  # construct col name
            data[colname] = data['T'].shift(i)
            base += var.replace('T', f'Th_{i}')  # add Th terms

    if dmax > 0:
        # create lagged moving avg temperature
        for i in range(dmax):
            colname = 'Td_' + str(i + 1)  # construct col name
            data[colname] = data['T'].shift(1 + i * 24).rolling(window=24, min_periods=24).mean()
            base += var.replace('T', f'Td_{i + 1}')  # add Td terms

    return data, base  # here returns updated base


def add_sync_sin_days(data: pd.DataFrame, fit_test=3, use_eq=True, hmax=0, dmax=0, start=69, end=100, stepsize=10,
                      sin1t='VE', sin2t='VE', sin3t=False, sin4t=False, sin05t='VE', sin025t=False, abs1t=False,
                      show_bm_perf=False, bm_model='Vanilla', plot_pred=False, plot_date='2013-03-26',
                      return_data=False, show_CV=False, clipneg=False, NCEMC_data=True, incl_mon=False,
                      RH1t=False, RH2t=False, RH05t=False, WS1t=False, WS2t=False, WS05t=False):
    """
    Sync the start dates for all sin waves. Shift days. "Sine wave start day=shift day(i) + 1"
    sin1t: 一年1周期, can be ['VE', 'AE'...], 'VE', 'AE', 'False'. False: don't include sin1t.
    sin2t: 一年2周期
    ...sinnt: 一年n周期
    sin05t: 两年1周期, 在一年上就是半个周期的弧形
    abs1t: sin1t的绝对值
    use_eq: True: use equinox, False: CV select start date
    bm_model: the model used for benchmark, can be 'Vanilla', 'RH', and 'WS'
    clipneg: whether to clip negative values for RH and WS models
    NCEMC_data: NCEMC one has RH and WS. ISONE only have Temperature.
    incl_mon: 试month和sin同时加入model
    show_CV: use_eq=True时是否算用equinox时的CV，默认为False（因为不用选参），但是就是跟data driven (必算CV)的CV结果做个对比而已
    hmax, dmax: recency h and d. Max here does not mean anything.
    """
    global result_df, subset_yr, j  # result_df to store results, j循环数，用来改method名字
    data = data.copy()  # make a copy such that function won't modify input df
    yr_list = np.unique(data.year)
    T_term = '+ C(hour) * T + C(hour) * I(T**2) + C(hour) * I(T**3) '
    f = 'load ~ trend + C(hour) * C(weekday) ' + T_term
    sin_T_term = '+ sin * T + sin * I(T**2) + sin * I(T**3) '

    # method = 'day|'  # initialize, indicate it is sine by day
    method = ''

    if incl_mon:
        mon_term = '+ C(month) * T + C(month) * I(T**2) + C(month) * I(T**3) '
        f += mon_term
        method += 'ms'  # month and sine
        if RH1t:
            RH_term = '+ RHS * T + RHS * I(T**2) + I(RHS**2) * T + I(RHS**2) * I(T**2) + C(hour) * RHS + C(hour) * I(RHS**2) '
            f += RH_term
            method += '+ RH '
        if WS1t:
            WS_term = '+ WSS * T + WSS * C(hour) '
            f += WS_term
            method += '+ WS'

    sin_RH_T = '+ RHsin * T + RHsin * I(T**2) + RH2sin * T + RH2sin * I(T**2) + '
    sin_RH_Hour = '+ RHsin * C(hour) + RH2sin * C(hour) '
    sin_RH = sin_RH_T + sin_RH_Hour

    sin_WS_T = '+ WSsin * T '
    sin_WS_Hour = '+ WSsin * C(hour) '
    sin_WS = sin_WS_T + sin_WS_Hour

    if show_bm_perf:  # show only, not returning anything
        benchmark_perf(data=data, fit_test=fit_test, bm_model=bm_model, hmax=hmax, dmax=dmax, show_CV=show_CV)

    if use_eq:  # use equinox days to shift, which are stored in dictionary

        method += sin1t  # use sin1t for method name

        global ve_day_dict, ss_day_dict, ae_day_dict, ws_day_dict

        for year in yr_list:
            # data.loc[data.year == year, 'new_dayofyr'] = np.roll(data.loc[data.year == year, 'dayofyr'],
            #                                                      (eq_day_dict[year] - 1) * 24)
            data.loc[data.year == year, 've_dayofyr'] = np.roll(data.loc[data.year == year, 'dayofyr'],
                                                                (ve_day_dict[year] - 1) * 24)
            data.loc[data.year == year, 'ss_dayofyr'] = np.roll(data.loc[data.year == year, 'dayofyr'],
                                                                (ss_day_dict[year] - 1) * 24)
            data.loc[data.year == year, 'ae_dayofyr'] = np.roll(data.loc[data.year == year, 'dayofyr'],
                                                                (ae_day_dict[year] - 1) * 24)
            data.loc[data.year == year, 'ws_dayofyr'] = np.roll(data.loc[data.year == year, 'dayofyr'],
                                                                (ws_day_dict[year] - 1) * 24)
        if sin1t:
            # Need to come back and tweak the name for RH and WS terms
            w = 1  # w is the factor of sin(wt)
            if 'VE' in sin1t:
                name = 've1t'
                method += f'-{name}'
                data.loc[:, name] = np.sin((w * 2 * pi / data.daysinyr) * data.ve_dayofyr)
                f += sin_T_term.replace('sin', name)
            if 'Ssol' in sin1t:
                name = 'ss1t'
                method += f'-{name}'
                data.loc[:, name] = np.sin((w * 2 * pi / data.daysinyr) * data.ss_dayofyr)
                f += sin_T_term.replace('sin', name)
            if 'AE' in sin1t:
                name = 'ae1t'
                method += f'-{name}'
                data.loc[:, name] = np.sin((w * 2 * pi / data.daysinyr) * data.ae_dayofyr)
                f += sin_T_term.replace('sin', name)
            if 'Wsol' in sin1t:
                name = 'ws1t'  # this is winter solstice, not wind speed
                method += f'-{name}'
                data.loc[:, name] = np.sin((w * 2 * pi / data.daysinyr) * data.ws_dayofyr)
                f += sin_T_term.replace('sin', name)

            # -- !!! Be aware, only one name (ve1t/ss1t/ae1t/ws1t) will be passed to the code below --#
            if RH1t:
                f += sin_RH.replace('sin', name)
                method += '-RH1t'
                if clipneg:
                    data.loc[:, 'RH' + name] = data['RH'] * data[name].clip(lower=0)
                    data.loc[:, 'RH2' + name] = data['RH'] ** 2 * data[name].clip(lower=0)
                else:
                    data.loc[:, 'RH' + name] = data['RH'] * data[name]
                    data.loc[:, 'RH2' + name] = data['RH'] ** 2 * data[name]
            if WS1t:
                f += sin_WS.replace('sin', name)
                method += '-WS1t'
                if clipneg:
                    data.loc[:, 'WS' + name] = data['WS'] ** 0.16 * data[name].clip(lower=0)
                else:
                    data.loc[:, 'WS' + name] = data['WS'] ** 0.16 * data[name]

        if abs1t:  # abs of ve1t, not quite working, worse than ve1t
            name, w = 'abs1t', 1
            method += f'-{name}'
            data.loc[:, name] = np.abs(np.sin((w * 2 * pi / data.daysinyr) * data.ve_dayofyr))
            f += sin_T_term.replace('sin', name)

        if sin2t:
            w = 2
            if 'VE' in sin2t:
                name = 've2t'
                method += f'-{name}'
                data.loc[:, name] = np.sin((w * 2 * pi / data.daysinyr) * data.ve_dayofyr)
                f += sin_T_term.replace('sin', name)
            if 'Ssol' in sin2t:
                name = 'ss2t'
                method += f'-{name}'
                data.loc[:, name] = np.sin((w * 2 * pi / data.daysinyr) * data.ss_dayofyr)
                f += sin_T_term.replace('sin', name)
            if 'AE' in sin2t:
                name = 'ae2t'
                method += f'-{name}'
                data.loc[:, name] = np.sin((w * 2 * pi / data.daysinyr) * data.ae_dayofyr)
                f += sin_T_term.replace('sin', name)
            if 'Wsol' in sin2t:
                name = 'ws2t'  # this is winter solstice, not wind speed
                method += f'-{name}'
                data.loc[:, name] = np.sin((w * 2 * pi / data.daysinyr) * data.ws_dayofyr)
                f += sin_T_term.replace('sin', name)

            if RH2t:
                f += sin_RH.replace('sin', name)
                method += '-RH2t'
                if clipneg:
                    data.loc[:, 'RH' + name] = data['RH'] * data[name].clip(lower=0)
                    data.loc[:, 'RH2' + name] = data['RH'] ** 2 * data[name].clip(lower=0)
                else:
                    data.loc[:, 'RH' + name] = data['RH'] * data[name]
                    data.loc[:, 'RH2' + name] = data['RH'] ** 2 * data[name]
            if WS2t:
                f += sin_WS.replace('sin', name)
                method += '-WS2t'
                if clipneg:
                    data.loc[:, 'WS' + name] = data['WS'] ** 0.16 * data[name].clip(lower=0)
                else:
                    data.loc[:, 'WS' + name] = data['WS'] ** 0.16 * data[name]

        if sin3t:
            name, w = 'sin3t', 3
            f += sin_T_term.replace('sin', name)
            data.loc[:, name] = np.sin((w * 2 * pi / data.daysinyr) * data.ve_dayofyr)
        if sin4t:
            name, w = 'sin4t', 4
            f += sin_T_term.replace('sin', name)
            data.loc[:, name] = np.sin((w * 2 * pi / data.daysinyr) * data.ve_dayofyr)
        if sin05t:
            w = 0.5
            if 'VE' in sin05t:
                name = 've05t'
                method += f'-{name}'
                data.loc[:, name] = np.sin((w * 2 * pi / data.daysinyr) * data.ve_dayofyr)
                f += sin_T_term.replace('sin', name)
            if 'Ssol' in sin05t:
                name = 'ss05t'
                method += f'-{name}'
                data.loc[:, name] = np.sin((w * 2 * pi / data.daysinyr) * data.ss_dayofyr)
                f += sin_T_term.replace('sin', name)
            if 'AE' in sin05t:
                name = 'ae05t'
                method += f'-{name}'
                data.loc[:, name] = np.sin((w * 2 * pi / data.daysinyr) * data.ae_dayofyr)
                f += sin_T_term.replace('sin', name)
            if 'Wsol' in sin05t:
                name = 'ws05t'  # this is winter solstice, not wind speed
                method += f'-{name}'
                data.loc[:, name] = np.sin((w * 2 * pi / data.daysinyr) * data.ws_dayofyr)
                f += sin_T_term.replace('sin', name)

            if RH05t:
                f += sin_RH.replace('sin', name)
                method += '-RH05t'
                data.loc[:, 'RH' + name] = data['RH'] * data[name]
                data.loc[:, 'RH2' + name] = data['RH'] ** 2 * data[name]
            if WS05t:
                f += sin_WS.replace('sin', name)
                method += '-WS05t'
                data.loc[:, 'WS' + name] = data['WS'] ** 0.16 * data[name]

        if sin025t:
            name, w = 'sin025t', 0.25
            f += sin_T_term.replace('sin', name)
            data.loc[:, name] = np.sin((w * 2 * pi / data.daysinyr) * data.ve_dayofyr)

        # method += f'|{j}'  # 在末端放入j，就是shift start deg，便于排序

        print(f)  # 打印f
        if hmax + dmax > 0:  # if either hmax or dmax >0, execute recency
            data, f = recency(data, f, hmax, dmax)
            print(f'Recency:{f}')  # 打印recency f

        # do CV and show performance
        if show_CV:
            train = data[data.year != yr_list[-1]].copy()
            mape = []
            for year in np.unique(train.year):  # Conduct CV
                tr = train[train.year != year]  # use previous 2 years as training set
                val = train[train.year == year]  # val set
                y_tr, X_tr = patsy.dmatrices(f, tr, return_type='dataframe')  # training
                y_val, X_val = patsy.dmatrices(f, val, return_type='dataframe')  # validation, is on training + val set
                model = sm.OLS(y_tr, X_tr).fit()  # use training period to fit model
                mape.append(MAPE(y_val, model.predict(X_val)))
            print(f'shift day using dictionary, Average MAPE across {len(yr_list) - 1}-fold CV: {np.mean(mape):.3f}')
            # 有算cv时再存入result_df
            result_df = result_df.append(
                {'year': subset_yr[-1], 'type': 0, 'method': method + f'|h{hmax}d{dmax}', 'mape': np.mean(mape),
                 'params': len(model.params), 'shift_deg': j}, ignore_index=True)

        # Evaluate on test set
        print(f'Test set performance use recent {fit_test} years for param estimation.')
        y_tr, X_tr = patsy.dmatrices(f, data[data.year.isin(yr_list[-fit_test - 1:-1])], return_type='dataframe')
        y_test, X_test = patsy.dmatrices(f, data[data.year == yr_list[-1]], return_type='dataframe')

        # time_start = time.time() # examine the training time
        model = sm.OLS(y_tr, X_tr).fit()  # use whole training period to fit model
        # print(f'Training time={time.time() - time_start:.2f} seconds.')

        print(f'No. of non-zero parameters: {len(model.params)}')
        mape_test = MAPE(y_test, model.predict(X_test))
        print(f'shift day using dictionary, Test MAPE: {mape_test:.3f}')
        print(f'---------------------------------------------------------------------------')

        # 下面存放test performance
        result_df = result_df.append(
            {'year': subset_yr[-1], 'type': fit_test, 'method': method + f'|h{hmax}d{dmax}', 'mape': mape_test,
             'params': len(model.params), 'shift_deg': j}, ignore_index=True)

    else:  # if not using equinox, search on start days. 以下sin1t如果用默认值'VE', 也是达到了True的效果
        method += 'Data-driven'
        mape_list = []  # store mape of each shift day
        names_ws = []  # store the chosen sine wave names and w's

        # initialize search list
        # search_list = list(range(start, end, stepsize)) + list(range(start + 250, end + 250, stepsize))  # by days
        # search_list = [64, 79, 94, 109] + [307, 322, 338, 353]  # by 15 deg, for NCEMC
        search_list = [64, 79, 94] + [155, 170] + [165, 180] + [322, 338, 353]  # by 15 deg, for NCEMC
        # search_list = [109, 125, 140, 155] + [307, 322, 338, 353]  # by 15 deg, for ISONE
        # search_list = [3, 18, 33, 49, 64, 79, 94, 109, 125, 140, 155, 170, 185, 201, 216, 231, 246, 262, 277, 292,
        #                307, 322, 338, 353]  # for DD CV illustration
        for i in search_list:
            # use specified shift day and update new_dayofyr col in data
            for year in yr_list:
                data.loc[data.year == year, 'new_dayofyr'] = np.roll(data.loc[data.year == year, 'dayofyr'], i * 24)

            if sin1t:
                name, w = 'sin1t', 1  # w is the factor of sin(wt)
                data.loc[:, name] = np.sin((w * 2 * pi / data.daysinyr) * data.new_dayofyr)
                if i == search_list[0]:  # execute only once
                    f += sin_T_term.replace('sin', name)
                if RH1t:
                    if i == search_list[0]:  # execute only once
                        f += sin_RH.replace('sin', name)
                        method += '-RH1t'
                    if clipneg:
                        data.loc[:, 'RH' + name] = data['RH'] * data[name].clip(lower=0)
                        data.loc[:, 'RH2' + name] = data['RH'] ** 2 * data[name].clip(lower=0)
                    else:
                        data.loc[:, 'RH' + name] = data['RH'] * data[name]
                        data.loc[:, 'RH2' + name] = data['RH'] ** 2 * data[name]
                if WS1t:
                    if i == search_list[0]:  # execute only once
                        f += sin_WS.replace('sin', name)
                        method += '-WS1t'
                    if clipneg:
                        data.loc[:, 'WS' + name] = data['WS'] ** 0.16 * data[name].clip(lower=0)
                    else:
                        data.loc[:, 'WS' + name] = data['WS'] ** 0.16 * data[name]

                names_ws.append((name, w))

            if sin2t:
                name, w = 'sin2t', 2
                data.loc[:, name] = np.sin((w * 2 * pi / data.daysinyr) * data.new_dayofyr)
                if i == search_list[0]:  # execute only once
                    f += sin_T_term.replace('sin', name)
                if RH2t:
                    if i == search_list[0]:  # execute only once
                        f += sin_RH.replace('sin', name)
                        method += '-RH2t'
                    if clipneg:
                        data.loc[:, 'RH' + name] = data['RH'] * data[name].clip(lower=0)
                        data.loc[:, 'RH2' + name] = data['RH'] ** 2 * data[name].clip(lower=0)
                    else:
                        data.loc[:, 'RH' + name] = data['RH'] * data[name]
                        data.loc[:, 'RH2' + name] = data['RH'] ** 2 * data[name]
                if WS2t:
                    if i == search_list[0]:  # execute only once
                        f += sin_WS.replace('sin', name)
                        method += '-WS2t'
                    if clipneg:
                        data.loc[:, 'WS' + name] = data['WS'] ** 0.16 * data[name].clip(lower=0)
                    else:
                        data.loc[:, 'WS' + name] = data['WS'] ** 0.16 * data[name]

                names_ws.append((name, w))

            if sin3t:
                name, w = 'sin3t', 3
                if i == search_list[0]:  # execute only once
                    f += sin_T_term.replace('sin', name)  # update formula w/ extra sin2t terms
                data.loc[:, name] = np.sin((w * 2 * pi / data.daysinyr) * data.new_dayofyr)

                names_ws.append((name, w))

            if sin4t:
                name, w = 'sin4t', 4
                if i == search_list[0]:  # execute only once
                    f += sin_T_term.replace('sin', name)  # update formula w/ extra sin2t terms
                data.loc[:, name] = np.sin((w * 2 * pi / data.daysinyr) * data.new_dayofyr)

                names_ws.append((name, w))

            if sin05t:
                name, w = 'sin05t', 0.5
                data.loc[:, name] = np.sin((w * 2 * pi / data.daysinyr) * data.new_dayofyr)
                if i == search_list[0]:  # execute only once
                    f += sin_T_term.replace('sin', name)
                if RH05t:
                    f += sin_RH.replace('sin', name)
                    if i == search_list[0]:  # execute only once
                        method += '-RH05t'
                    data.loc[:, 'RH' + name] = data['RH'] * data[name]
                    data.loc[:, 'RH2' + name] = data['RH'] ** 2 * data[name]
                if WS05t:
                    f += sin_WS.replace('sin', name)
                    if i == search_list[0]:  # execute only once
                        method += '-WS05t'
                    data.loc[:, 'WS' + name] = data['WS'] ** 0.16 * data[name]

                names_ws.append((name, w))

            if sin025t:
                name, w = 'sin025t', .25
                if i == search_list[0]:  # execute only once
                    f += sin_T_term.replace('sin', name)  # update formula w/ extra sin2t terms
                data.loc[:, name] = np.sin((w * 2 * pi / data.daysinyr) * data.new_dayofyr)

                names_ws.append((name, w))

            # recency需要且只用执行一次。recency创建新的Th, Td, 跟sine无关。f公式也只用更新一次，这点与前面相同。
            if i == search_list[0]:  # execute only once
                print(f)
                if hmax + dmax > 0:  # if either hmax or dmax >0, execute recency
                    data, f = recency(data, f, hmax, dmax)
                    print(f'Recency:{f}')

            train = data[data.year != yr_list[-1]].copy()
            mape = []

            for year in np.unique(train.year):  # Conduct CV
                tr = train[train.year != year]  # use previous 2 years as training set
                val = train[train.year == year]  # val set
                y_tr, X_tr = patsy.dmatrices(f, tr, return_type='dataframe')  # training
                y_val, X_val = patsy.dmatrices(f, val, return_type='dataframe')  # validation, is on training + val set
                model = sm.OLS(y_tr, X_tr).fit()  # use training period to fit model
                mape.append(MAPE(y_val, model.predict(X_val)))
            mape_list.append((i, np.mean(mape)))  # store (shift_day, mean_cv_mape)
            print(f'shift day={i}, Average MAPE across {len(yr_list) - 1}-fold CV: {np.mean(mape):.3f}')

        mape_list.sort(key=lambda tup: tup[1])  # sort ascending by the second column: cv mape
        best_day, best_cv_mape = mape_list[0][0], mape_list[0][1]
        print(f'-----Best shift day={best_day}, {len(yr_list) - 1}-fold CV MAPE={best_cv_mape:.3f}-----')

        # update data with best shift day
        for year in yr_list:
            data.loc[data.year == year, 'new_dayofyr'] = np.roll(data.loc[data.year == year, 'dayofyr'], best_day * 24)

        # update all existing sine cols with best_day, then update RH and WS cols accordingly
        for (name, w) in names_ws:
            data.loc[:, name] = np.sin((w * 2 * pi / data.daysinyr) * data.new_dayofyr)

            if clipneg:
                data.loc[:, 'RH' + name] = data['RH'] * data[name].clip(lower=0)
                data.loc[:, 'RH2' + name] = data['RH'] ** 2 * data[name].clip(lower=0)
                data.loc[:, 'WS' + name] = data['WS'] ** 0.16 * data[name].clip(lower=0)
            elif NCEMC_data:  # process if using NCEMC data. NCEMC one has RH and WS. ISONE only have Temperature.
                data.loc[:, 'RH' + name] = data['RH'] * data[name]
                data.loc[:, 'RH2' + name] = data['RH'] ** 2 * data[name]
                data.loc[:, 'WS' + name] = data['WS'] ** 0.16 * data[name]
            else:  # pass when using ISONE data
                pass

        print(f'Test set performance use recent {fit_test} years for param estimation.')
        y_tr, X_tr = patsy.dmatrices(f, data[data.year.isin(yr_list[-fit_test - 1:-1])], return_type='dataframe')
        y_test, X_test = patsy.dmatrices(f, data[data.year == yr_list[-1]], return_type='dataframe')
        model = sm.OLS(y_tr, X_tr).fit()  # use whole training period to fit model
        mape_test = MAPE(y_test, model.predict(X_test))
        print(f'Best shift day={best_day}, Test MAPE: {mape_test:.3f}')
        print(f'---------------------------------------------------------------------------')

        # return best_day, best_cv_mape, mape_test  # return 3 items, including the selected shift day
        result_df = result_df.append(
            {'year': subset_yr[-1], 'type': 0, 'method': method + f'|h{hmax}d{dmax}', 'mape': best_cv_mape,
             'SD': best_day,
             'params': len(model.params)}, ignore_index=True)
        result_df = result_df.append(
            {'year': subset_yr[-1], 'type': fit_test, 'method': method + f'|h{hmax}d{dmax}', 'mape': mape_test,
             'SD': best_day,
             'params': len(model.params)}, ignore_index=True)

    if plot_pred:
        # plt.figure(figsize=(12, 5))
        # plt.plot(model.predict(X_test), label='predict')
        # plt.plot(y_test, label='actual')
        # plt.legend()
        # plt.show()

        # use date as index for plot, create a new index "date"
        test = data[data.year == yr_list[-1]]
        base = datetime(test.year.unique()[0], 1, 1)
        date = np.array([base + timedelta(hours=i) for i in range(1, len(test) + 1)])
        data_plot = pd.concat([y_test, model.predict(X_test)], axis=1).reset_index(drop=True)
        data_plot.rename(columns={'load': 'Actual', 0: 'Predict'}, inplace=True)
        data_plot = data_plot.set_index(date)

        # extract yr, month, day
        plot_date = list(map(int, plot_date.split('-')))
        date_from = datetime(plot_date[0], plot_date[1], plot_date[2])
        date_to = date_from + timedelta(days=7)

        plt.plot(data_plot.Actual.loc[date_from:date_to], color='black', label='Actual')  # plot actual, use index
        plt.plot(data_plot.Predict[date_from:date_to], marker='o', linestyle='--', label=f'Sine model prediction')

        # remove dup legend, 2 Actuals created by 2 loops
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='best')

        plt.title(f'Performance Comparison for a week starting {plot_date}')

        plt.xlabel('Date')
        plt.ylabel('Load (kW)')

    if return_data:
        test = data[data.year == yr_list[-1]]
        test['predict'] = model.predict(X_test)
        return test
    else:
        return None


def wss(data, process_weather=False, weather_var=('RH', 'WS'), f=None, tot_sta_num=18,
        wss_list=None, idx=None):
    """
    Weather station selection.
    Parameters
    ----------
    data: input dataframe, with at least temperature
    process_weather: True, use the wss result to combine other weather variables in the input dataset. Require input to have other vars.
    weather_var: indicate the other weather variables
    f: choose model to conduct wss. Below we use Vanilla model.
    tot_sta_num: total no. of stations
    use_wss_list: True - use the wss_list from txt file. False - conduct wss, and store results in wss_list
    idx: the index to be used from wss_list

    Returns:
    dataframe with "T" as the temperature of selected stations
    -------

    """
    # global wss_list
    if wss_list is None:  # if None, need to conduct wss.
        wss_list = []
        wss_flag = 'Conduct WSS'  # create a local var
        # step 1, fit each weather station to the training period, record in-sample fit MAPE, default training length=2 yrs
        years = np.unique(data.year)
        tr = data[data.year.isin(years[:2])]
        mape = []
        T_term = '+ C(month) * T + C(month) * I(T**2) + C(month) * I(T**3) + C(hour)*T + C(hour) * I(T**2) + C(hour) * I(T**3)'
        for i in range(1, tot_sta_num + 1):
            f = 'load ~ trend + C(hour) * C(weekday) + C(month) ' + T_term.replace('T', f'T{i}')  # use Vanilla model
            y_tr, X_tr = patsy.dmatrices(f, tr, return_type='dataframe')  # training
            model = sm.OLS(y_tr, X_tr).fit()  # use training period to fit model
            mape.append([f'T{i}', MAPE(y_tr, model.predict(X_tr))])

        mape.sort(key=lambda x: x[1])  # sort ascending by the second column: validation mape
        sta_sorted = list(np.array(mape)[:, 0])  # get sorted stations

        # step 2, combine top k stations and forecast on validation data, record MAPE
        mape_comb_sta = []
        f = 'load ~ trend + C(hour) * C(weekday) + C(month) ' + T_term  # reset model
        for k in range(1, tot_sta_num + 1):  # k=2: combine top 2 stations
            data['T'] = data[sta_sorted[:k]].mean(axis=1)
            tr, val = data[data.year.isin(years[:2])], data[data.year == years[2]]

            y_tr, X_tr = patsy.dmatrices(f, tr, return_type='dataframe')  # training
            y_val, X_val = patsy.dmatrices(f, val, return_type='dataframe')  # validation
            model = sm.OLS(y_tr, X_tr).fit()  # use training period to fit model
            mape_comb_sta.append([k, MAPE(y_val, model.predict(X_val))])

        mape_comb_sta.sort(key=lambda x: x[1])  # sort ascending by validation mape
        k_sel = mape_comb_sta[0][0]  # get the selected top k

        print(f'Top {k_sel} stations selected, station list={sta_sorted[:k_sel]}')
        wss_list.append(sta_sorted[:k_sel])  # store wss result in global list

        data['T'] = data[sta_sorted[:k_sel]].mean(axis=1)


    else:  # wss_list is not None, use wss_list, update T col
        wss_flag = 'Use Existing WSS'
        data = data.copy()  # make a copy such that function won't modify input df
        data['T'] = data[wss_list[idx]].mean(axis=1)

    data.drop(['T' + str(i) for i in range(1, tot_sta_num + 1)], axis=1, inplace=True)  # keep T col only

    if process_weather:
        for var in weather_var:
            if wss_flag == 'Use Existing WSS':
                sel_var = [sub.replace('T', var) for sub in wss_list[idx]]  # replace names before
            else:
                sel_var = [sub.replace('T', var) for sub in sta_sorted[:k_sel]]  # replace names before
            data[var] = data[sel_var].mean(axis=1)
            data.drop([var + str(i) for i in range(1, tot_sta_num + 1)], axis=1, inplace=True)  # keep mean var only

    return data


def shift_solar_terms(shiftDeg, input_df=ve):
    """
    input_df: ve, ssol, ae, wsol
    shift dict to get correlated solar terms. 从已知equinox和solstice推导其他solar terms日期。图方便。
    shiftDeg can be positive or negative.
    """
    input_dict = dict(zip(input_df.year, input_df.dayofyr))  # 设回原点
    input_dict.update((x, int(round(y + shiftDeg / 360 * 365))) for x, y in input_dict.items())

    return input_dict
