import logging
import requests
import pandas as pd
import os
import numpy as np

import datetime as dt

TODAY = dt.date.today()


class FMP(object):
    def __init__(self, api_key):
        self.APIKey = api_key

    def getFMPData(self, url):
        if url.find("?") >= 0:
            operator = "&"
        else:
            operator = "?"
        logging.info(f"executing request: {url}{operator}apikey={self.APIKey}")
        r_object = requests.get(
            f'{url}{operator}apikey={self.APIKey}')
        try:
            r_object = r_object.json()
        except:
            r_object = {}
        return r_object

    def screenSymbols(self, marketcap=50, excludedIndustries=["Utilities", "Financial Services"], includedExchanges=["Amsterdam",  "Brussels",  "EURONEXT",  "Irish",  "Lisbon",  "MCE",  "Paris",  "TSXV",  "XETRA"], use_filter=False):
        querytickers = self.getFMPData(
            f"https://financialmodelingprep.com/api/v3/stock-screener?marketCapMoreThan={marketcap*1000000}")

        if use_filter == True:
            for ticker in querytickers[:]:
                if ticker["sector"] in excludedIndustries:
                    querytickers.remove(ticker)
                elif ticker["exchange"] not in includedExchanges:
                    querytickers.remove(ticker)
                    #print(f"removed {ticker}")
        return querytickers

    def getReturns(self, stocks, start_date, end_date=TODAY, trailingStopLoss=1.0):
        assets = pd.DataFrame()
        all_stocks = stocks
        if len(stocks) > 5:
            # split into chunks of 5
            stocks_chunks = [stocks[i:i + 5] for i in range(0, len(stocks), 5)]
        else:
            stocks_chunks = [stocks]
        for stocks in stocks_chunks:
            stocks_str = ",".join(stocks)
            stock_data = self.getFMPData(f'https://financialmodelingprep.com/api/v3/historical-price-full/{stocks_str}?from={start_date}&to={end_date}')
            # check if stock_data contains a key "historicalStockList"
            if "historicalStockList" in stock_data:
                stock_data = stock_data["historicalStockList"]
            else:
                stock_data = [stock_data]
            for i in range(len(stock_data)):
                # create a dataframe from the historical data
                df = pd.DataFrame(stock_data[i]["historical"])
                # set the index to the date
                df.set_index("date", inplace=True)
                # drop all columns except adjClose
                df = df[["adjClose"]]
                # rename the column to the ticker
                df.columns = [stock_data[i]["symbol"]]
                # order returns by date
                df = df.sort_index()
                # apply trailing stop loss
                df = self._trailingStopLoss(df, stop=trailingStopLoss)
                # append the dataframe to the assets dataframe
                assets = assets.join(df, how="outer")
        assets = assets.filter(all_stocks)
        ret_data = assets.pct_change()[1:]
        # FMP API does not return a column for a stock that does not have data.
        # determine number of columns in ret_data
        num_cols = len(ret_data.columns)
        wts = np.array([1/num_cols]*num_cols)
        returns = (ret_data * wts).sum(axis=1)
        # format date in returns as date
        returns.index = pd.to_datetime(returns.index)
        return returns.squeeze()

    def _trailingStopLoss(self, data, stop=1.0):
        # implements a trailing stop loss
        sold = False
        lastStop = 0
        for i in range(len(data)):
            if data.iloc[i,0] < lastStop:
                sold = True
            if sold == True:
                data.iloc[i,0] = lastStop
            elif data.iloc[i,0]*(1-stop) > lastStop:
                lastStop = data.iloc[i,0]*(1-stop)
        return data