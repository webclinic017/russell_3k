import streamlit as st
from datetime import datetime, date, timedelta
import pandas as pd
import yfinance as yf
from pathlib import Path
from os.path import exists
from yahooquery import Ticker


class Recommendations2(object):


    def __init__(self, today):  # Variables
        self.today_stamp = today
        self.saveMonth = str(today)[:7]
        self.saveDay = str(today)[8:10]

        self.saveRec = Path(f"data/recommenders/{str(self.today_stamp)[:4]}/{self.saveMonth}/{self.today_stamp}/")
        if not self.saveRec.exists():
            self.saveRec.mkdir(parents=True)

        self.saveRaw = Path(f"data/raw/{self.saveMonth}/{self.today_stamp}/")
        if not self.saveRaw.exists():
            self.saveRaw.mkdir(parents=True)

        self.saveScreeners = Path(f"data/screeners/{self.saveMonth}/{self.today_stamp}/")
        if not self.saveScreeners.exists():
            self.saveScreeners.mkdir(parents=True)

        self.saveTickers = Path(f"data/tickers/{self.saveMonth}/{self.today_stamp}/")
        if not self.saveTickers.exists():
            self.saveTickers.mkdir(parents=True)


    def run_rec2(self):
        if exists(self.saveRec / f"recommender_02_return_dataFrame.pkl"):
            return
        else:
            self.tickers = list(pd.read_pickle(self.saveRec / f"recommender_01_return_dataFrame.pkl")["Symbol"])
            self.sName = "Recommender 02 Return List"
            self.index_name = "^GSPC"  # S&P 500
            self.start_date = datetime.now() - timedelta(days=365)
            self.today = self.today_stamp
            self.end_date = date.today()

        returns_multiples = []
        exportList = pd.DataFrame(
            columns=[
                "Symbol",
                "RS_Rating",
                "Returns_multiple",
                "Current_Price",
                "20 Day MA",
                "50 Day Ma",
                "200 Day MA",
                "52 Week Low",
                "52 week High",
            ]
        )

        # Index Returns
        index_df = yf.download("^GSPC", period="1y")
        index_df["Percent Change"] = index_df["Close"].pct_change()
        index_return = (index_df["Percent Change"] + 1).cumprod()[-1]

        # Find top 30% performing Tickers (relative to the S&P 500)
        tickers = Ticker(
            self.tickers, 
            # asynchronous=True
        )
        df3 = tickers.history(period="1y")
        df3.columns = [e.title() for e in df3.columns]
        c0 = 0
        for s in self.tickers:
            c0 += 1
            one = df3.T[s]
            df = one.T
            df.to_pickle(self.saveRaw / f"{s}.pkl")

            # Calculating returns relative to the market (returns multiple)
            try:
                df["Percent Change"] = df["Close"].pct_change()
                stock_return = (df["Percent Change"] + 1).cumprod()[-1]
                returns_multiple = round((stock_return / index_return), 2)
                returns_multiples.extend([returns_multiple])
                print(f"{c0}) Ticker: {s}; Returns Multiple against S&P 500: {returns_multiple}\n")
            except Exception:
                pass

        # Creating dataframe of only top 31%
        rs_df = pd.DataFrame(list(zip(self.tickers, returns_multiples)), columns=["Symbol", "Returns_multiple"])
        rs_df["RS_Rating"] = rs_df["Returns_multiple"].rank(pct=True) * 100
        rs_df = rs_df[rs_df.RS_Rating >= rs_df.RS_Rating.quantile(0.69)]

        # Checking Minervini conditions of top 30% of stocks in given list
        rs_stocks = rs_df["Symbol"]
        for stock in rs_stocks:
            try:
                df = pd.read_pickle(self.saveRaw / f"{stock}.pkl")
                sma = [20, 50, 200]
                for x in sma:
                    df["SMA_" + str(x)] = round(df["Close"].rolling(window=x).mean(), 2)
                # Storing required values
                Current_Price = df["Close"][-1]
                moving_average_20 = df["SMA_20"][-1]
                moving_average_50 = df["SMA_50"][-1]
                moving_average_200 = df["SMA_200"][-1]
                low_of_52week = round(min(df["Low"][-260:]), 2)
                high_of_52week = round(max(df["High"][-260:]), 2)
                RS_Rating = round(rs_df[rs_df["Symbol"] == stock].RS_Rating.tolist()[0])
                Returns_multiple = rs_df[rs_df["Symbol"] == stock].Returns_multiple.tolist()[0]
                try:
                    moving_average_200_20 = df["SMA_200"][-20]
                except Exception:
                    moving_average_200_20 = 0

                # -> Condition_1 :: [Current Price > 50_SMA > 200_SMA]
                condition_1 = Current_Price > moving_average_50 > moving_average_200
                # -> Condition_2 :: [50 SMA > 200 SMA]
                condition_2 = moving_average_50 > moving_average_200
                # -> Condition_3 :: [200 SMA trending up for at least 1 month]
                condition_3 = moving_average_200 > moving_average_200_20
                # -> Condition_4 :: [20_SMA > 50_SMA >_200_SMA]
                condition_4 = moving_average_20 > moving_average_50 > moving_average_200
                # -> Condition_5 :: [200 SMA trending up over last 1month]
                condition_5 = moving_average_200 > moving_average_200_20
                # -> Condition_6 :: [Current Price is at least 10% above 52 week low]
                condition_6 = Current_Price >= (1.1 * low_of_52week)
                # -> Condition_7 :: [Current Price is equal to or higher than 50% of 52 week high]
                condition_7 = Current_Price >= (0.50 * high_of_52week)

                # If all conditions above are true, add Ticker to exportList
                if (
                    condition_1
                    & condition_2
                    & condition_3
                    & condition_4
                    & condition_5
                    & condition_6
                    & condition_7
                ):
                    exportList = exportList.append(
                        {
                            "Symbol": stock,
                            "RS_Rating": RS_Rating,
                            "Returns_multiple": Returns_multiple,
                            "Current_Price": Current_Price,
                            "20 Day MA": moving_average_20,
                            "50 Day Ma": moving_average_50,
                            "200 Day MA": moving_average_200,
                            "52 Week Low": low_of_52week,
                            "52 week High": high_of_52week,
                        },
                        ignore_index=True,
                    ).sort_values(by="RS_Rating", ascending=False)

            except Exception as e:
                print(f"{e} - Could not gather data on {stock}")

        exportList = exportList.drop_duplicates(subset="Symbol")
        exportList["rank"] = range(1, len(exportList["Symbol"]) + 1)
        exportList = exportList.set_index("rank")
        exportList.to_pickle(self.saveRec / "recommender_02_return_dataFrame.pkl")
        return


    def run_rec2_personal_port(self, start_lst):
        self.tickers = start_lst
        self.sName = "Recommender 02 Return List"
        self.index_name = "^GSPC"  # S&P 500
        self.start_date = datetime.now() - timedelta(days=365)
        self.today = self.today_stamp
        self.end_date = date.today()
        returns_multiples = []
        exportList = pd.DataFrame(
            columns=[
                "Symbol",
                "RS_Rating",
                "Returns_multiple",
                "Current_Price",
                "20 Day MA",
                "50 Day Ma",
                "200 Day MA",
                "52 Week Low",
                "52 week High",
            ]
        )

        # Index Returns
        index_df = yf.download("^GSPC", period="1y")
        index_df["Percent Change"] = index_df["Close"].pct_change()
        index_return = (index_df["Percent Change"] + 1).cumprod()[-1]

        # Find top 30% performing Tickers (relative to the S&P 500)
        c0 = 0
        for ticker in self.tickers:
            c0 += 1
            try:
                df = yf.download(ticker, period="1y")
                df.to_pickle(f"data/bunker/{ticker}.pkl")
            except Exception:
                pass

            # Calculating returns relative to the market (returns multiple)
            try:
                df["Percent Change"] = df["Close"].pct_change()
                stock_return = (df["Percent Change"] + 1).cumprod()[-1]
                returns_multiple = round((stock_return / index_return), 2)
                returns_multiples.extend([returns_multiple])
                st.write(f"{c0}) Ticker: {ticker}; Returns Multiple against S&P 500: {returns_multiple}\n")
            except Exception:
                pass

        # Creating dataframe of only top 30%
        rs_df = pd.DataFrame(list(zip(self.tickers, returns_multiples)), columns=["Symbol", "Returns_multiple"])
        rs_df["RS_Rating"] = rs_df.Returns_multiple.rank(pct=True) * 100
        rs_df = rs_df[rs_df.RS_Rating >= rs_df.RS_Rating.quantile(0.65)]

        # Checking Minervini conditions of top 30% of stocks in given list
        rs_stocks = rs_df["Symbol"]
        for stock in rs_stocks:
            try:
                df = pd.read_pickle(f"data/bunker/{stock}.pkl")
                sma = [20, 50, 200]
                for x in sma:
                    df["SMA_" + str(x)] = round(df["Close"].rolling(window=x).mean(), 2)

                # Storing required values
                Current_Price = df["Close"][-1]
                moving_average_20 = df["SMA_20"][-1]
                moving_average_50 = df["SMA_50"][-1]
                moving_average_200 = df["SMA_200"][-1]
                low_of_52week = round(min(df["Low"][-260:]), 2)
                high_of_52week = round(max(df["High"][-260:]), 2)
                RS_Rating = round(rs_df[rs_df["Symbol"] == stock].RS_Rating.tolist()[0])
                Returns_multiple = rs_df[rs_df["Symbol"] == stock].Returns_multiple.tolist()[0]
                try:
                    moving_average_200_20 = df["SMA_200"][-20]
                except Exception:
                    moving_average_200_20 = 0

                # # Condition 1: Current Price > 50 SMA and > 200 SMA
                # condition_1 = Current_Price > moving_average_50 > moving_average_200
                # # Condition 2: 50 SMA and > 200 SMA
                # condition_2 = moving_average_50 > moving_average_200
                # # Condition 3: 200 SMA trending up for at least 1 month
                # condition_3 = moving_average_200 > moving_average_200_20
                # # Condition 4: 50 SMA>50 SMA and 50 SMA> 200 SMA
                # condition_4 = moving_average_20 > moving_average_50 > moving_average_200
                # # Condition 5: Current Price > 50 SMA
                # condition_5 = Current_Price > moving_average_50
                # # Condition 6: Current Price is at least 10% above 52 week low
                # condition_6 = Current_Price >= (1.1 * low_of_52week)
                # # Condition 7: Current Price is within 50% of 52 week high
                # condition_7 = Current_Price >= (0.50 * high_of_52week)

                # -> Condition_1 :: [Current Price > 50_SMA > 200_SMA]
                condition_1 = Current_Price > moving_average_50 > moving_average_200
                # -> Condition_2 :: [50 SMA > 200 SMA]
                condition_2 = moving_average_50 > moving_average_200
                # -> Condition_3 :: [200 SMA trending up for at least 1 month]
                condition_3 = moving_average_200 > moving_average_200_20
                # -> Condition_4 :: [20_SMA > 50_SMA >_200_SMA]
                condition_4 = moving_average_20 > moving_average_50 > moving_average_200
                # -> Condition_5 :: [200 SMA trending up over last 1month]
                condition_5 = moving_average_200 > moving_average_200_20
                # -> Condition_6 :: [Current Price is at least 10% above 52 week low]
                condition_6 = Current_Price >= (1.1 * low_of_52week)
                # -> Condition_7 :: [Current Price is equal to or higher than 50% of 52 week high]
                condition_7 = Current_Price >= (0.50 * high_of_52week)

                # If all conditions above are true, add Ticker to exportList
                if (
                    condition_1
                    & condition_2
                    & condition_3
                    & condition_4
                    & condition_5
                    & condition_6
                    & condition_7
                ):
                    exportList = exportList.append(
                        {
                            "Symbol": stock,
                            "RS_Rating": RS_Rating,
                            "Returns_multiple": Returns_multiple,
                            "Current_Price": Current_Price,
                            "20 Day MA": moving_average_20,
                            "50 Day Ma": moving_average_50,
                            "200 Day MA": moving_average_200,
                            "52 Week Low": low_of_52week,
                            "52 week High": high_of_52week,
                        },
                        ignore_index=True,
                    ).sort_values(by="RS_Rating", ascending=False)

            except Exception as e:
                print(f"{e} - Could not gather data on {stock}")

        exportList = exportList.drop_duplicates(subset="Symbol")
        exportList["rank"] = range(1, len(exportList["Symbol"]) + 1)
        return exportList


if __name__ == "__main__":
    today="2021-09-21"
    today_stamp=str(today)[:10]
    month_stamp=str(today_stamp)[:7]
    year_stamp=str(today_stamp)[:4]

    initial_loc = "/home/gdp/4m/data/recommenders/"
    source = "recommender_01_return_dataFrame"

    filler = f"{initial_loc}/{year_stamp}/{month_stamp}/{today_stamp}/{source}.pkl"
    ticker_lst = list(pd.read_pickle(filler)["Symbol"])

    Recommendations2(today_stamp).run_rec2()