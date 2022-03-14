import warnings
from ast import Pass

from pages import strategy

warnings.filterwarnings("ignore")
import time
from datetime import date, datetime, timedelta
from os.path import exists
from pathlib import Path

import numpy as np
import pandas as pd
import pickle5 as pickle
import src.data.yahoo_fin_stock_info as si
import src.models.portfolio.proof as p1
import src.models.portfolio.proof_port as p2
import src.models.strategy as s1
import streamlit as st
import yfinance as yf
from src.models.portfolio.web_monteCarloCholesky import \
    MonteCarloCholesky as mcc
from src.models.portfolio.web_pca import The_PCA_Analysis as pca
from yahooquery import Ticker

import pages as p0


def clean(list1):
    temp = []
    for i in list1:
        if type(i) == int:
            temp.append(float(i))
        if type(i) == float:
            temp.append(i)
        elif type(i)==str:
            temp.append(float(i[:-1]))
    return temp           


def clean2(list1):
    temp = []
    for i in list1:
        if type(i) == int:
            temp.append(float(i))
        if type(i) == float:
            temp.append(i)
        elif type(i)==str:
            temp.append(float(i[1:]))
    return temp     




class Proof_69(object):
    
    def __init__(self, day):            
        self.starter_date = str(day)[:10]
        self.ender_date = str(datetime.now())[:10]     
        
        self.day = str(day)[:10]
        self.month = str(self.day)[:7]
        self.year = str(self.day)[:4]
        
        self.saveSP500 = Path(f"data/russell_3k/{self.year}/{self.month}/{self.day}/")
        if not self.saveSP500.exists():
            self.saveSP500.mkdir(parents=True)               

        self.reports_portfolio = Path(f"reports/portfolio/{self.month}/{self.day}/")
        if not self.reports_portfolio.exists():
            self.reports_portfolio.mkdir(parents=True)            
            
        self.reports_portResults = Path(f"reports/port_results/{self.month}/{self.starter_date}/")
        if not self.reports_portResults.exists():
            self.reports_portResults.mkdir(parents=True)                               


    def model(self):
        st.sidebar.header("__[3] Inputs__")
        use_pca = 'Yes'
        use_indicators = 'Yes'
        sma_ema_choice = 'SMA' # "EMA"
        run_list = ['monte_carlo']
        methodology = 'Option_None'                
        graph1 = 'No'                          
        file_saver = 'Yes'
        indicator_method = 'Double SMA' # ['Single SMA', 'Double SMA', 'Bollinger Bands', 'MACD', 'RSI']
               
        initial_investment = 2500.0
        ret_multiple = .75
        min_composite_score = 50.0
        num_portfolios = 21000
        max_allocations = 25.0
        min_sentiment_score = 1.0
        min_analyst_recom_score = 60.0
        min_rs_rating_score = 40.0
        y_factor = 0.66
        low_52wk = 0.75                              
              
        p999 = (f"data/russell_3k/{self.year}/{self.month}/{self.day}/russell_3k_recommender.csv")    
        if exists(p999):
            data_1 = pd.read_csv(p999)
            
        else:
            russell_1k_stocks = si.tickers_russell1000()
            russell_2k_stocks = list(pd.read_csv("data/ticker_lists/russell_2k_stock_lst.csv")['Ticker'])
            russell_3k_stocks = russell_1k_stocks + russell_2k_stocks
            data_1 = pd.DataFrame(russell_3k_stocks)
            data_1.to_csv(p999)
            

        
            
        data = data_1.copy()         
        data.columns = [x.lower() for x in data.columns]
        data.columns = [x.replace(' ', '_') for x in data.columns]
        data.columns = [x.replace('(', '') for x in data.columns]
        data.columns = [x.replace(')', '') for x in data.columns]
        data.columns = [x.replace('/', '') for x in data.columns]
        data.columns = [x.replace('-', '') for x in data.columns]
        data.columns = [x.replace('.', '') for x in data.columns]
        data = pd.DataFrame(data).fillna(0.0).round(2)                    
            
        try:
            del data['earnings_date']
            del data['ipo_date']
        except Exception:
            pass

        try:                    
            data = data.rename(columns={'symbol': 'ticker'})
        except Exception:
            pass
        
        try:               
            data = data.rename(columns={'52_week_low': 'low_52_week'})
            data = data.rename(columns={'52_week_high': 'high_52_week'})
        except Exception:
            pass
        
        try:                
            data = data.rename(columns={'relative_strength': 'rs_rating'})
            data = data.rename(columns={'adj_analyst_recom': 'analyst_recom'})
        except Exception:
            pass
                
        try:
            del data['index']
        except Exception:
            pass       
        
        try:                
            del data['rank']
        except Exception:
            pass       
        
        try:
            data['low_52_week'] = clean2(data['low_52_week'])
            data['high_52_week'] = clean2(data['high_52_week'])                                    
        except Exception:
            pass
            
            data['over_52wk_low'] = (data['current_price'] / data['low_52_week'])            


        data = data.query(
            f"my_score >= {min_composite_score} \
                and rs_rating >= {min_rs_rating_score} \
                    and sentiment_score >= {min_sentiment_score} \
                        and analyst_recom >= {min_analyst_recom_score} \
                            and returns_multiple >= {ret_multiple} \
                                and current_price >= low_52_week * {low_52wk}"
        )   
            
            
        self.recommender_dataset = pd.DataFrame(data.copy()).fillna(0.0).round(2)
        port_tics = list(self.recommender_dataset["ticker"])       


        hammerTime = Ticker(
            port_tics,
            asynchronous=True,
            formatted=False,
            backoff_factor=0.34,
            progress=True,
            validate=True,
            verify=True,
        )
        hammer_hist = hammerTime.history(period='1y').reset_index().set_index('date')
        hammer_hist.index = pd.to_datetime(hammer_hist.index)
        hammer_hist = hammer_hist.rename(columns={'symbol': 'ticker'})
        bulk_files = pd.DataFrame()
        for i in port_tics:
            try:
                z = pd.DataFrame(hammer_hist[hammer_hist['ticker'] == i]['adjclose'])
                bulk_files[i] = z
            except:
                port_tics.remove(i)                 
        bulk_files = bulk_files.dropna(axis='columns')
        bulk_files.index = pd.to_datetime(bulk_files.index)  
        
        bulk_files = pd.DataFrame(bulk_files.reindex(sorted(bulk_files.columns), axis=1).copy())
        df_train_data = bulk_files.loc[:self.starter_date]
        df_test_data = pd.DataFrame(bulk_files.loc[self.starter_date:].copy())
        
        bulk_files.to_pickle(self.saveSP500 / f"{methodology}.pkl")
        df_train_data.to_pickle(self.saveSP500 / f"df_train_data.pkl")
        df_test_data.to_pickle(self.saveSP500 / f"df_test_data.pkl") 
        


        if graph1 == 'Yes':
            graph101 = True
        elif graph1 == 'No':
            graph101 = False
            
        if file_saver == 'Yes':
            return_files = True
        elif file_saver == 'No':
            return_files = False                

        chicken_dinna = pca(
            port_tics, 
            self.starter_date, 
            return_files, 
            y_factor
        ).build_pca(df_train_data, graph101)  
        
        df_pca_output = df_train_data.filter(chicken_dinna)

        saver_lst_1 = p2.The_Portfolio_Optimizer(
            self.starter_date, 
            return_files,
        ).optimize(
            initial_investment, 
            num_portfolios, 
            max_allocations, 
            df_pca_output,
            graph101,
        )
                
                
        crap_lst = []
        sheen_lst = []
        
        from src.models.strategy.indicators import Indicator_Ike as ii

        def sharper(df1):
            port_tics = list(df1["ticker"])
            c, cc, ccc = 0.0, 0.0, len(port_tics)

            for p in port_tics:                        
                if p in crap_lst or p in sheen_lst:
                    pass

                else:
                    temp = pd.DataFrame(hammer_hist.loc[:self.starter_date])
                    data = pd.DataFrame(temp[temp['ticker'] == p])
                    try:
                        del data['ticker']
                        del data['dividends']                      
                    except Exception:
                        pass
                    try:
                        if indicator_method == 'Single SMA':
                            cc += 1
                            x = p0.Strategy(self.starter_date).run_optimal_sma(p, data, graph101, cc, ccc)
                            if x is True:
                                c += 1
                                sheen_lst.append(p)
                            else:
                                crap_lst.append(p)
                                df1 = df1.drop(df1[df1['ticker'] == p].index)

                        if indicator_method == 'Double SMA':                                
                            cc += 1
                            x = p0.Strategy(self.starter_date).run_movAvg_sma_ema(p, data, sma_ema_choice, graph101, cc, ccc)
                            
                            if x is True:
                                c += 1
                                sheen_lst.append(p)
                            else:
                                crap_lst.append(p)
                                df1 = df1.drop(df1[df1['ticker'] == p].index)
                                
                        if indicator_method == 'Bollinger Bands':
                            cc += 1
                            x = ii(p, str(self.starter_date)[:10], cc, ccc, graph101).kingpin(indicator_method)
                            if x == p:
                                c += 1
                                sheen_lst.append(p)
                            else:
                                crap_lst.append(p)
                                df1 = df1.drop(df1[df1['ticker'] == p].index) 
                                
                        if indicator_method == 'MACD':
                            cc += 1
                            x = ii(p, str(self.starter_date)[:10], cc, ccc, graph101).kingpin(indicator_method)
                            if x == p:
                                c += 1
                                sheen_lst.append(p)
                            else:
                                crap_lst.append(p)
                                df1 = df1.drop(df1[df1['ticker'] == p].index) 
                                
                        if indicator_method == 'RSI':
                            cc += 1
                            x = ii(p, str(self.starter_date)[:10], cc, ccc, graph101).kingpin(indicator_method)
                            if x == p:
                                c += 1
                                sheen_lst.append(p)
                            else:
                                crap_lst.append(p)
                                df1 = df1.drop(df1[df1['ticker'] == p].index) 
                                                                                                                                                        
                    except:
                        crap_lst.append(p)
                        df1 = df1.drop(df1[df1['ticker'] == p].index)
                time.sleep(1.3)
            return df1



        if 'monte_carlo' in run_list:
            try:
                monte_carlo_cholesky = pd.DataFrame(saver_lst_1[5]).reset_index()
            except:
                path1 = (f"reports/portfolio/{str(self.starter_date)[:7]}/{str(self.starter_date)[:10]}/max_sharpe_df_3.pkl")
                with open(path1, "rb" ) as fh:
                    monte_carlo_cholesky = pickle.load(fh)

            try:
                monte_carlo_cholesky = monte_carlo_cholesky.rename(columns={'symbol': 'ticker'})
            except Exception:
                pass    

            if use_indicators == 'Yes':
                monte_carlo_cholesky = sharper(monte_carlo_cholesky)                    
            else:
                monte_carlo_cholesky = pd.DataFrame(monte_carlo_cholesky.copy())                    

            monte_carlo_cholesky = list(set(monte_carlo_cholesky['ticker']))
            num_portfolios=3400
            fin = mcc(self.starter_date).montecarlo_sharpe_optimal_portfolio(monte_carlo_cholesky, num_portfolios, self.starter_date)
            fin.columns = ["ticker", "allocation"]

            data = pd.DataFrame(df_test_data)
            data = data.filter(fin['ticker'])                             
            p1.Proof_of_Concept(self.starter_date, self.ender_date, return_files, True).setup(fin.reset_index(), "monte_carlo_cholesky", data, initial_investment)
            
            
        return
