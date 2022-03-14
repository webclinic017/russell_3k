import streamlit as st
import warnings
from bs4 import BeautifulSoup
import time
from urllib.request import urlopen, Request
import nltk
nltk.download("vader_lexicon")
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import yfinance as yf
from pathlib import Path
from os.path import exists
from yahooquery import Ticker
import numpy as np
from finviz import get_news
import pickle5 as pickle



class Simple_Rick(object):

    def __init__(self, today_stamp):  # Variables
        today_stamp = str(today_stamp)[:10]
        month_stamp = str(today_stamp)[:7]
        year_stamp = str(today_stamp)[:4]
        self.saveFinviz = Path(f"data/finviz/{month_stamp}/{today_stamp}/finviz.csv")
        self.saveRec = Path(f"data/recommenders/{year_stamp}/{month_stamp}/{today_stamp}/")
        self.saveRaw = Path(f"data/raw/{month_stamp}/{today_stamp}/")
        self.path1 = Path(f"data/sentiment/{year_stamp}/{month_stamp}/{today_stamp}/")
        if not self.path1.exists():
            self.path1.mkdir(parents=True)        



    def run_rec_0(self):
        if not exists(self.saveFinviz):
            st.write("no file found")
            st.write(str(self.saveFinviz))
            return

        if exists(self.saveFinviz):
            df = pd.DataFrame(pd.read_csv(self.saveFinviz))

            df = df.rename(columns={'20-Day Simple Moving Average': 'Simple Moving Average 20-Day'}) 
            df = df.rename(columns={'50-Day Simple Moving Average': 'Simple Moving Average 50-Day'}) 
            df = df.rename(columns={'200-Day Simple Moving Average': 'Simple Moving Average 200-Day'}) 
            df = df.rename(columns={'50-Day High': 'High 50-Day'}) 
            df = df.rename(columns={'50-Day Low': 'Low 50-Day'}) 
            df = df.rename(columns={'52-Week High': 'High 52-Week'}) 
            df = df.rename(columns={'52-Week Low': 'Low 52-Week'})

            df.columns = [x.lower() for x in df.columns]
            df.columns = [x.replace(' ', '_') for x in df.columns]
            df.columns = [x.replace('(', '') for x in df.columns]
            df.columns = [x.replace(')', '') for x in df.columns]
            df.columns = [x.replace('/', '') for x in df.columns]
            df.columns = [x.replace('-', '') for x in df.columns]
            df.columns = [x.replace('.', '') for x in df.columns]

            df.to_pickle(self.saveRec / "recommender_00_return_dataFrame.pkl")
            symbol_list = list(df["ticker"])
            st.write(f" * [X] [COMPLETE] - Recommender Stage #00 - [Total Passed == {len(symbol_list)}]","\n",)



    def run_rec_1(self):
        if exists(self.saveRec / "recommender_01_return_dataFrame.pkl"):
            path1 = Path(self.saveRec / "recommender_01_return_dataFrame.pkl")
            with open(path1, "rb") as fh:
                data = pickle.load(fh)
                st.write(f" * [X] [COMPLETE] - Recommender Stage #01 - [Total Passed == {len(data['ticker'])}]","\n",)
                return

        elif exists(self.saveRec / "recommender_00_return_dataFrame.pkl"):
            path1 = Path(self.saveRec / "recommender_00_return_dataFrame.pkl")
            with open(path1, 'rb') as fh:
                rec_00_df = pickle.load(fh)
                rec_01_df = pd.DataFrame(rec_00_df)

        rec_01_df = rec_01_df[rec_01_df["analyst_recom"] < 2.6]
        rec_01_df.to_pickle(self.saveRec / "recommender_01_return_dataFrame.pkl")
        st.write(f" * [X] [COMPLETE] - Recommender Stage #01 - [Total Passed == {len(rec_01_df['ticker'])}]","\n",)
        return



    def run_rec_2(self):
        if exists(self.saveRec / f"recommender_02_return_dataFrame.pkl"):
            path1 = Path(self.saveRec / f"recommender_02_return_dataFrame.pkl")
            with open(path1, "rb") as fh:
                data = pickle.load(fh)
                rec_02_tickers = list(data["ticker"])
                st.write(f" * [X] [COMPLETE] - Recommender Stage #02 - [Total Passed == {len(rec_02_tickers)}]","\n",)
                return

        elif exists(self.saveRec / "recommender_01_return_dataFrame.pkl"):
            path1 = Path(self.saveRec / "recommender_01_return_dataFrame.pkl")
            with open(path1, 'rb') as fh:
                data = pickle.load(fh)
                rec_02_tickers = list(data["ticker"])
                
        returns_multiples = []
        exportList = pd.DataFrame(
            columns=[
                "ticker",
                "rs_rating",
                "returns_multiple",
                "current_price",
                "ma_20_Day",
                "ma_50_Day",
                "ma_200_Day",
                "low_52_Week",
                "high_52_week",
            ]
        )

        # Index Returns
        index_df = yf.download("^GSPC", period="1y")
        index_df["pct_change"] = index_df["Close"].pct_change()
        index_return = (index_df["pct_change"] + 1).cumprod()[-1]

        # Find top 30% performing Tickers (relative to the S&P 500)
        tickers = Ticker(
            rec_02_tickers,
            asynchronous=True,
            formatted=False,
            backoff_factor=0.34,
            validate=True,
        )

        df_0 = tickers.history(period="1y")
        df_0.columns = [e.title() for e in df_0.columns]
        c0 = 0

        for s in rec_02_tickers:
            try:
                c0 += 1
                df_1 = df_0.T[s]
                df = pd.DataFrame(df_1.T)
                df.to_pickle(self.saveRaw / f"{s}.pkl")
            except:
                rec_02_tickers.remove(s)

            # Calculating returns relative to the market (returns multiple)
            try:
                df["pct_change"] = df["Close"].pct_change()
                stock_return = (df["pct_change"] + 1).cumprod()[-1]
                returns_multiple = round((stock_return / index_return), 2)
                returns_multiples.extend([returns_multiple])
                # st.write(f"{c0}) Ticker: {s}; Returns Multiple against S&P 500: {returns_multiple}\n")
            except Exception:
                pass

        # Creating dataframe of only top 31%
        rs_df = pd.DataFrame(list(zip(rec_02_tickers, returns_multiples)),columns=["ticker", "returns_multiple"],)
        rs_df["rs_rating"] = rs_df["returns_multiple"].rank(pct=True) * 100
        rs_df = rs_df[rs_df["rs_rating"] >= rs_df["rs_rating"].quantile(0.69)]

        # Checking Minervini conditions of top 31% of stocks in given list
        rs_stocks = rs_df["ticker"]
        for stock in rs_stocks:
            try:
                df = pd.DataFrame(pd.read_pickle(self.saveRaw / f"{stock}.pkl"))
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
                RS_Rating = round(rs_df[rs_df["ticker"] == stock].rs_rating.tolist()[0])
                Returns_multiple = rs_df[rs_df["ticker"] == stock].returns_multiple.tolist()[0]
                try:
                    moving_average_200_20 = df["SMA_200"][-20]
                except Exception:
                    moving_average_200_20 = 0

                # -> Condition_1 :: [Current Price > 50_SMA > 200_SMA]
                condition_1 = Current_Price > moving_average_50 > moving_average_200

                # -> Condition_2 :: [20_SMA > 50_SMA >_200_SMA]
                condition_2 = moving_average_20 > moving_average_50 > moving_average_200
                
                # -> Condition_3 :: [200 SMA trending up for at least 1 month]
                condition_3 = moving_average_200 > moving_average_200_20               
                
                # -> Condition_4 :: [Current Price is at least 10% above 52 week low]
                condition_4 = Current_Price >= (1.34 * low_of_52week)
                
                # -> Condition_5 :: [Current Price is equal to or higher than 50% of 52 week high]
                condition_5 = Current_Price >= (0.69 * high_of_52week)
                

                # If all conditions above are true, add Ticker to exportList
                if (
                    condition_1
                    & condition_2
                    & condition_3
                    & condition_4
                    & condition_5
                ):
                    exportList = exportList.append(
                        {
                            "ticker": stock,
                            "rs_rating": RS_Rating,
                            "returns_multiple": Returns_multiple,
                            "current_price": Current_Price,
                            "ma_20_Day": moving_average_20,
                            "ma_50_Day": moving_average_50,
                            "ma_200_Day": moving_average_200,
                            "low_52_Week": low_of_52week,
                            "high_52_week": high_of_52week,
                        },
                        ignore_index=True,
                    ).sort_values(by="rs_rating", ascending=False)            

            except Exception as e:
                st.write(f"{e} - Could not gather data on {stock}")

        exportList = exportList.drop_duplicates(subset="ticker")
        exportList.to_pickle(self.saveRec / "recommender_02_return_dataFrame.pkl")
        st.write(f" * [X] [COMPLETE] - Recommender Stage #02 - [Total Passed == {len(exportList['ticker'])}]","\n",)
        return



    def run_rec_3(self):
        if exists(self.saveRec / f"recommender_03_return_dataFrame.pkl"):
            rec_03_tickers = list(pd.read_pickle(self.saveRec / f"recommender_02_return_dataFrame.pkl")["ticker"])
            st.write(f" * [X] [COMPLETE] - Recommender Stage #03 - [Total Passed == {len(rec_03_tickers)}]","\n",)
            return

        elif not exists(self.saveRec / f"recommender_02_return_dataFrame.pkl"):
            st.write("* * * [ERROR] NO FILE FOUND FOR RECOMMENDER 1 * * *")
            return

        rec_03_tickers = list(pd.read_pickle(self.saveRec / f"recommender_02_return_dataFrame.pkl")["ticker"])


        def sentiment_analysis(newS, stocks):
            for stock in stocks:
                dates = newS['Date']
                headlines = newS['News Headline']
                links = newS['Article Link']
                sources = newS['Source']
                parsed_news=[]
                for r in range(len(newS)):
                    parsed_news.append([stock, dates[r], headlines[r], links[r], sources[r]])    

                # Sentiment Analysis
                analyzer = SentimentIntensityAnalyzer()
                columns = ["ticker", "Date", "News Headline", "Article Link", "Source"]
                news = pd.DataFrame(parsed_news, columns=columns)
                scores = news["News Headline"].apply(analyzer.polarity_scores).tolist()
                df_scores = pd.DataFrame(scores)
                news = news.join(df_scores, rsuffix="_right")       

                # View Data
                news["Date"] = pd.to_datetime(news.Date).dt.date
                unique_ticker = news["ticker"].unique().tolist()
                news_dict = {name: news.loc[news["ticker"] == name] for name in unique_ticker}
                values = []     

            for stock in stocks:
                dataframe = news_dict[stock]
                dataframe = dataframe.set_index("ticker")
                dataframe = dataframe.drop(columns=["News Headline"])
                mean = round(dataframe["compound"].mean() * 100, 0)
                values.append(mean)    

            df = pd.DataFrame(stocks, columns=["ticker"])
            df["sentiment_score"] = values
            return df


        def mini_news(stocks):
            for stock in stocks:
                print ('\nRecent News: ')
                print ('Getting data for ' + stock + '...\n')        
                if exists(self.path1 / f"df_single_news_{stock}.csv"):
                    pass
                else:
                    try:
                        data_news = get_news(stock)
                        df_single_news = pd.DataFrame((data_news), columns=['Date','News Headline','Article Link','Source'])
                        df_single_news.to_csv(self.path1 / f"df_single_news_{stock}.csv")
                        time.sleep(2.5)
                    except:
                        print(f'BAD TICKER {stock}')
                        stocks.remove(stock)
            return stocks            


        def run_sentiment(stocks):
            print(len(stocks))
            df = pd.DataFrame()
            symbols = []
            sentiments = []
            stocks = mini_news(stocks)
            print(len(stocks))
            for stock in stocks:
                if exists(self.path1 / f"{stock}_sentiment.csv"):
                    fd = pd.read_csv(self.path1 / f"{stock}_sentiment.csv")
                    symbols.append(fd['ticker'].loc[0])
                    sentiments.append(fd['sentiment_score'].loc[0])
                else:
                    newS = pd.read_csv(self.path1 / f"df_single_news_{stock}.csv")
                    fd = sentiment_analysis(newS, [stock])
                    symbols.append(fd['ticker'].loc[0])
                    sentiments.append(fd['sentiment_score'].loc[0])
                    fd.to_csv(self.path1 / f"{stock}_sentiment.csv")
                    time.sleep(2.5)
            df['ticker'] = symbols
            df['sentiment_score'] = sentiments
            return df
            
        df_final = run_sentiment(rec_03_tickers)
        df_final.to_pickle(self.saveRec / "recommender_03_return_dataFrame.pkl")
        st.write(f" * [X] [COMPLETE] - Recommender Stage #03 - [Total Passed == {len(df_final['ticker'])}]","\n",)
        return


    def run_rec_4(self):
        if exists(self.saveRec / "recommender_04_return_dataFrame.pkl"):
            rec_04 = pd.read_pickle(self.saveRec / "recommender_04_return_dataFrame.pkl")
            symbol_list = list(rec_04["ticker"])
            st.write(f" * [X] [COMPLETE] - Recommender Stage #04 - [Total Passed == {len(symbol_list)}]","\n",)
            return

        rec_final_01 = pd.read_pickle(self.saveRec / "recommender_01_return_dataFrame.pkl").round(2)
        rec_final_02 = pd.read_pickle(self.saveRec / "recommender_02_return_dataFrame.pkl").round(2)
        rec_final_03 = pd.read_pickle(self.saveRec / "recommender_03_return_dataFrame.pkl").round(2)

        rec_final_01.columns = [x.lower() for x in rec_final_01.columns]
        rec_final_02.columns = [x.lower() for x in rec_final_02.columns]
        rec_final_03.columns = [x.lower() for x in rec_final_03.columns]

        rec_final_01 = rec_final_01.rename(columns={'analyst_recom': 'ar'})
        analyst_recom = [2.6, 2.5, 2.4, 2.3, 2.2, 2.1, 2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0,]
        Analyst_Recom = list(np.arange(50.0, 101.0, 3.13).round())
        d1 = dict(zip(analyst_recom, Analyst_Recom))
        adj__analyst_lst = []
        for i in rec_final_01["ar"]:
            for key, val in d1.items():
                if i == key:
                    adj__analyst_lst.append(val)
        del rec_final_01["ar"]
        rec_final_01["analyst_recom"] = adj__analyst_lst


        rec_final_01 = rec_final_01[rec_final_01["ticker"].isin(list(rec_final_03["ticker"]))]
        rec_final_02 = rec_final_02[rec_final_02["ticker"].isin(list(rec_final_03["ticker"]))]

        a = rec_final_01.merge(rec_final_02, how="inner", on="ticker")
        b = a.merge(rec_final_03, how="inner", on="ticker")

        b["est_gl"] = round(((b["target_price"] - b["current_price"]) / b["current_price"]) * 100,2,)
        b = b[b["est_gl"] > 1.0]

        final_df = pd.DataFrame(b.copy()).set_index('no')

        final_df["my_score"] = (
            (final_df["sentiment_score"])
            + (final_df["analyst_recom"])
            + (final_df["rs_rating"])
            ) / 3

        st.dataframe(final_df)

        final_df.to_pickle(self.saveRec / "recommender_04_return_dataFrame.pkl")
        st.write(f" * [X] [COMPLETE] - Recommender Stage #04 - [Total Passed == {len(final_df['ticker'])}]","\n",)
        return



    def run_rec_5(self):
        if exists(self.saveRec / "recommender_05_return_dataFrame.pkl"):
            rec_05_tickers_lst = list(pd.read_pickle(self.saveRec / "recommender_05_return_dataFrame.pkl")["ticker"])
            st.write(f" * [X] [COMPLETE] - Recommender Stage #05 - [Total Passed == {len(rec_05_tickers_lst)}]","\n",)
            return

        fd = pd.DataFrame(pd.read_pickle(self.saveRec / "recommender_04_return_dataFrame.pkl")).reset_index()
        fd = fd[fd["my_score"] > 49.99]
        fd.to_pickle(self.saveRec / "recommender_05_return_dataFrame.pkl")
        st.write(f"[COMPLETE] - Recommender Stage #05 - [Total Passed == {len(fd['ticker'])}]","\n",)
        return