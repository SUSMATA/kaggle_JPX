from random import sample
import numpy as np
import pandas as pd
from tqdm import tqdm
import re

#日付を連続的に生成する。

from datetime import datetime
from datetime import timedelta

def daterange(_start, _end):
    """
    日付を連続的に生成するiterator
    
    Parameters
    ----------
    _start : datetime.datetime
        この日からスタートさせたい(iterにはこの日も含む)
    _end : datetime.datetime
        この日までiterに含めたい
        
    Returns
    ----------
    iteratorが返ります。
    
    """
    for n in range((_end - _start).days + 1):
        yield _start + timedelta(n)


def listedDate(codeList):
    """
    コードの上場日を返す
    
    Input
    ----------
    codeList : list
        調べたいcodeのリスト
        
    Return
    ----------
    ListedDate : dict
        codeListに対応した上場日
    """
    import requests
    from bs4 import BeautifulSoup
    import re

    url_base = 'https://profile.yahoo.co.jp/fundamental/'

    ListedDate = {}

    bugs = []

    for i, code in tqdm(enumerate(codeList)):
        url = url_base + str(code)
        r = requests.get(url)
        soup = BeautifulSoup(r.content, "html.parser")

        tarText = list(soup.html.body.center.find_all('div', attrs={'class': 'profile'})[0].div.find_all('div')[8])[7]
        tarText = str(list(tarText.find_all('tr'))[11])

        try:
            m = re.search(r'\d+年\d+月\d+日', tarText)
            ListedDate[int(code)] = datetime.strptime(m.group(), '%Y年%m月%d日')
        except AttributeError:
            m = re.search(r'\d+年\d+月', tarText)
            ListedDate[int(code)] = datetime.strptime(m.group(), '%Y年%m月')

    return ListedDate

def extractDF(c, date, start=datetime(2020,9,1), end=datetime(2020,10,30)):
    
    a = df_stock_prices.loc[df_stock_prices['SecuritiesCode']==c]
    return a.loc[(a['Date'] > start) & (a['Date'] < end)]




def test_stock_price_preprocessing_for_eval(test_stock_prices_df):
    """スコア計算用のstock_pricesの前処理"""
    
    test_stock_prices_df['Date'] = pd.to_datetime(test_stock_prices_df['Date'])
    test_stock_prices_df = test_stock_prices_df[['Date', 'SecuritiesCode', 'Target']]
    
    return test_stock_prices_df

def submission_preprocessin_for_eval(submission_df):
    """スコア計算用のsubmission_dfの前処理"""
    
    submission_df['Date'] = pd.to_datetime(submission_df['Date'])
    
    return submission_df

def calc_spread_return_per_day(_day_pred, _day_test, _weights=np.linspace(2,1,200)):
    """
    当該日付分だけ抽出した状態で、当該日付のR_dayを計算
    """
    weights_mean = _weights.mean()
    _day_test = _day_test.merge(right=_day_pred, on='SecuritiesCode')
    _day_test.sort_values('Rank', inplace=True)

    day_S_up = (_day_test[:200]['Target'].values * _weights).sum() / weights_mean
    day_S_down = (_day_test[-200:]['Target'].values[::-1] * _weights).sum() / weights_mean
    day_R = day_S_up - day_S_down
    return day_R


# #./data/supplemental_files/stock_prices.csvより、publicLeaderboard用のtestデータをinput

# test_stock_prices_df = pd.read_csv('./data/supplemental_files/stock_prices.csv')
# test_stock_prices_df = test_stock_price_preprocessing_for_eval(test_stock_prices_df)

def PL_nonTimeSeriesAPI(submission, test_df):
    """
    2021/12/6から2022/2/28の期間のscoreを計算する。
    （ただし、TimeSeriesAPIには則っていないので、逐次的には計算していない。）
    
    Inputs
    ----------
    submission : pd.DataFrame
        sample_submissionに則った形式のデータ
    
    return
    ----------
    score : float
        期間中のscore
    """

    def test_stock_price_preprocessing_for_eval(test_stock_prices_df):
        """スコア計算用のstock_pricesの前処理"""
        
        test_stock_prices_df['Date'] = pd.to_datetime(test_stock_prices_df['Date'])
        test_stock_prices_df = test_stock_prices_df[['Date', 'SecuritiesCode', 'Target']]
        
        return test_stock_prices_df
    
    def calc_spread_return_per_day(_day_pred, _day_test, _weights=np.linspace(2,1,200)):
        """
        当該日付分だけ抽出した状態で、当該日付のR_dayを計算
        """
        weights_mean = _weights.mean()
        _day_test = _day_test.merge(right=_day_pred, on='SecuritiesCode')
        _day_test.sort_values('Rank', inplace=True)

        day_S_up = (_day_test[:200]['Target'].values * _weights).sum() / weights_mean
        day_S_down = (_day_test[-200:]['Target'].values[::-1] * _weights).sum() / weights_mean
        day_R = day_S_up - day_S_down
        return day_R

    test_df = test_stock_price_preprocessing_for_eval(test_df)
    submission['Date'] = pd.to_datetime(submission['Date'])
    start = test_df['Date'].min()
    end = test_df['Date'].max()
    R = []
    for date in daterange(start, end):
        day_test = test_df.loc[test_df['Date'] == date].drop('Date', axis=1)
        day_pred = submission.loc[submission['Date'] == date].drop('Date', axis=1)
        if day_test.shape[0] == 0:
            continue

        R.append(calc_spread_return_per_day(day_pred, day_test))
    R = np.array(R)
    score = R.mean()/ R.std()
    
    return score

def elim_distuber_convert_float(df, col_idx):
    """
    check_disturberで邪魔文字の確認後、それをnp.nanに置き換え、
    列自体をfloat型に置き換える
    
    Input
    ----------
    このセルの上のコメントを入れれば良い。
    
    Return
    ----------
    None
        
    結果としては、全てのデータで、'-'が邪魔しているということがわかった。
    """

    for col_id in col_idx:
        for i, e in enumerate(df.iloc[:, col_id]):
            if type(e) == str:
                m = re.search('[0-9]+', e)
                if m == None:
                    df.iloc[i, col_id] = np.nan
            elif type(e) == float:
                continue
            else:
                df.iloc[i, col_id] = np.nan
        col = df.columns[col_id]
        df[col] = df[col].astype(float)
        
    return None

def stock_prices_preprocessing(df_stock_prices):
    """
    ・Open,High,Low,CloseのAdjustmentFactorによる計算し直し
    ・4値のnullは、closeの線形補完で埋める。
    ・Target=nanの削除 = 上場日前のデータの削除
    """

    df_stock_prices['Date'] = pd.to_datetime(df_stock_prices['Date'])

    codes = df_stock_prices['SecuritiesCode'].unique()

    #・Open,High,Low,CloseのAdjustmentFactorによる計算し直し
    for c in codes:
        code_df = df_stock_prices.loc[df_stock_prices['SecuritiesCode'] == c]
        code_AF = np.cumprod(code_df['AdjustmentFactor'][::-1].values)[::-1]
        
        for col in ['Open', 'High', 'Low', 'Close']:
            df_stock_prices.loc[df_stock_prices['SecuritiesCode'] == c, [f"mod{col}"]] = \
                df_stock_prices.loc[df_stock_prices['SecuritiesCode']==c, col] * code_AF

    #・2020年10月1日のopen,high,low,closeがnanのものは、前日の終値で穴埋め
    #2020年10月1日も線形補完でいい。
    # after20201001 = []
    # for c in tqdm(codes):
    #     try:
    #         if df_stock_prices.loc[
    #             (df_stock_prices['SecuritiesCode']==c) & (df_stock_prices['Date']==datetime(2020,10,1)), 'Close'
    #             ].isnull().bool():

    #             for col in ['modOpen', 'modHigh', 'modLow', 'modClose']:
    #                 df_stock_prices.loc[
    #                     (df_stock_prices['SecuritiesCode']==c) & (df_stock_prices['Date']==datetime(2020,10,1)), col
    #                     ] = \
    #                     df_stock_prices.loc[
    #                     (df_stock_prices['SecuritiesCode']==c) & (df_stock_prices['Date']==datetime(2020,9,30)), 'modClose'
    #                     ].values[0]
    #     except:
    #         after20201001.append(c)

    #その他の4値は、closeの線形補完で埋める。
    hasCloseNull = df_stock_prices.groupby(['SecuritiesCode']).agg(lambda x: x.isnull().sum())['modClose']
    hasCloseNull = hasCloseNull[hasCloseNull > 0]
    hasCloseNull.sort_values(inplace=True)

    for c in hasCloseNull.index:
        df_stock_prices.loc[df_stock_prices['SecuritiesCode']==c, 'modClose'] = \
    df_stock_prices.loc[df_stock_prices['SecuritiesCode']==c, 'Close'].interpolate()

    df_stock_prices.dropna(subset=['modClose'], axis=0, inplace=True)

    df_stock_prices[['modOpen', 'modHigh', 'modLow', 'modClose']] = \
    df_stock_prices[['modOpen', 'modHigh', 'modLow', 'modClose']].fillna(method='backfill')


    #Target=nanの削除 = 上場日前のデータの削除 
    df_stock_prices.dropna(subset=['Target'], inplace=True)

    return df_stock_prices[['Date', 'SecuritiesCode', 'Volume', 
        'ExpectedDividend', 'SupervisionFlag', 'Target', 
        'modOpen', 'modHigh', 'modLow', 'modClose']]
