#--------------------------------------------------------------------------------------------------------------------------------------------------------
### -> -> -> LIBRARY & PACKAGE IMPORT <- <- <- 
#--------------------------------------------------------------------------------------------------------------------------------------------------------

import warnings

from pages import strategy
warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
from datetime import date, datetime, timedelta
from pathlib import Path
from os.path import exists
import numpy as np
import pickle5 as pickle 
import yfinance as yf
from yahooquery import Ticker
import time

import src.models.strategy as s1
import pages as p0
import src.models.portfolio.proof as p1
import src.models.portfolio.proof_port as p2
from src.models.portfolio.web_monteCarloCholesky import MonteCarloCholesky as mcc
from src.models.portfolio.web_pca import The_PCA_Analysis as pca


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


#--------------------------------------------------------------------------------------------------------------------------------------------------------
### -> -> -> PROOF CLASS <- <- <- 
#--------------------------------------------------------------------------------------------------------------------------------------------------------


class Proof(object):

    def __init__(self):            
        pass
    

    def prove_it(self):
        st.sidebar.header("__[2] Select Proof Mode__")
        self.mode_select = st.sidebar.radio("", ("Model", "Daily_Update", "Quick_View"))
        st.sidebar.markdown(f"{'__'*25}")
        

        if self.mode_select == "Model":
            self.model()

        if self.mode_select == "Daily_Update":
            self.daily_update()

        if self.mode_select == "Quick_View":
            self.quick_view()         
            

#--------------------------------------------------------------------------------------------------------------------------------------------------------
### -> -> -> PROOF - [Quick_View] <- <- <- 
#--------------------------------------------------------------------------------------------------------------------------------------------------------


    def quick_view(self):
        st.sidebar.header("__[3] Recommender Date:__")

        self.day = str(st.sidebar.date_input(
            label="",
            value=date(2021, 7, 14),
            min_value=date(2021, 7, 14),
            max_value=datetime.now(),
            key="Derpy Day 1",
            ))[:10]
        self.month = self.day[:7]
        st.sidebar.markdown(f"{'__'*25}")

        new_name_lst = [
            'maximum_sharpe_ratio',
            'minimum_volatility_portfolio',
            'maximum_sharpe_equalWT',
            'monte_carlo_cholesky',  
        ]
        portfolio_option = st.sidebar.multiselect(
            label='Select Portfolio Options ', 
            options=new_name_lst,
            default=(
                # 'maximum_sharpe_ratio',
                # 'minimum_volatility_portfolio',
                # 'maximum_sharpe_equalWT',
                'monte_carlo_cholesky', 
            )
        )           

        if st.sidebar.button("Run Quick_View"):
            st.title("__View Configured Portfolios__")
            st.header(f" PORTFOLIO RECOMMENDED DATE : [{self.day}]")
            st.write(f"{'__'*25} \n {'__'*25}")            

            for n in portfolio_option:
                try:
                    df = pd.read_csv(f"/home/gdp/russell_3k/reports/port_results/{self.month}/{self.day}/{n}.csv")    
                    try:
                        del df['Unnamed: 0']
                    except Exception:
                        pass
                    st.header(f"{n}")
                    st.dataframe(df.set_index(['companyName','ticker']))
                    st.write('__'*25)

                except:
                    st.write(f"{'__'*25}\n{'__'*25}")
                    st.subheader(" > NO DATA FOR THIS DATE - PLEASE CHOOSE A DIFFERENT DATE")
                    st.write(f"{'__'*25}\n{'__'*25}")        


#--------------------------------------------------------------------------------------------------------------------------------------------------------
### -> -> -> PROOF - [Daily_Update] <- <- <- 
#--------------------------------------------------------------------------------------------------------------------------------------------------------

    def daily_update(self):
        st.sidebar.header("__[3] Recommender Date:__")

        self.day = str(st.sidebar.date_input(
            label=" ",
            value=date(2021, 7, 14),
            min_value=date(2021, 7, 14),
            max_value=datetime.now(),
            key="Derpy Day 1",
        ))[:10]
        
        self.saveMonth = self.day[:7]
        self.saveReport = Path(f"reports/portfolio/{self.saveMonth}/{self.day}/")
        self.saveRec = Path(f"data/recommenders/{str(self.day)[:4]}/{self.saveMonth}/{self.day}/")
        self.saveRaw = Path(f"data/raw/{self.saveMonth}/{self.day}/")
        self.saveScreeners = Path(f"data/screeners/{self.saveMonth}/{self.day}/")
        self.saveTickers = Path(f"data/tickers/{self.saveMonth}/{self.day}/")        
        
        self.month = self.day[:7]
        st.sidebar.markdown(f"{'__'*25}")
        
        file_saver=st.sidebar.radio("save output?", ('No','Yes'))
        if file_saver == 'Yes':
            return_files = True
        elif file_saver == 'No':
            return_files = False        

        graph1 = st.sidebar.radio('graph it', ('True', 'False'))
        if graph1 == 'True':
            graph1 = True
        elif graph1 == 'False':
            graph1 = False

        new_name_lst = [
            'maximum_sharpe',
            'minimum_volatility',
            'equalWT',
            'monteCarloC', 
        ]
        portfolio_option = st.sidebar.multiselect(
            label='Select Portfolio Option', 
            options=new_name_lst,
            default=(
                # 'maximum_sharpe',
                # 'minimum_volatility',
                # 'equalWT',
                'monteCarloC', 
            )
        )                  

        st.sidebar.header("__[4] RUN:__")
        if st.sidebar.button("Run Daily_Update"):
            if exists(self.saveRec / f"recommender_05_return_dataFrame.pkl"):
                try:
                    with open(str(self.saveRec) + "/recommender_05_return_dataFrame.pkl", "rb" ) as fh:
                        self.recommender_dataset = pickle.load(fh)       

                    st.title("__View Configured Portfolios__")
                    st.header(f" PORTFOLIO RECOMMENDED DATE : [{self.day}]")
                    st.write(f"{'__'*25} \n {'__'*25}")
                    
                except:
                    st.write(f"{'__'*25}\n{'__'*25}")
                    st.subheader(" > NO DATA FOR THIS DATE - PLEASE CHOOSE A DIFFERENT DATE")
                    st.write(f"{'__'*25}\n{'__'*25}")

                p1.Proof_of_Concept_Viewer(self.day, initial_cash=2500, save_output=return_files, graphit=graph1).setup(portfolio_option)
                st.sidebar.write("__" * 25)
                     
            elif not exists(self.saveRec / f"recommender_05_return_dataFrame.pkl"):
                st.write(f"{'__'*25}\n{'__'*25}")
                st.subheader(" > NO DATA FOR THIS DATE - PLEASE CHOOSE A DIFFERENT DATE")
                st.write(f"{'__'*25}\n{'__'*25}")


#--------------------------------------------------------------------------------------------------------------------------------------------------------
### -> -> -> PROOF - [Model] <- <- <- 
#--------------------------------------------------------------------------------------------------------------------------------------------------------


    def model(self):
        st.sidebar.header("__[3] Inputs__")
        use_pca = 'Yes'
        use_indicators = 'Yes'
        sma_ema_choice = 'SMA' # "EMA"
        run_list = ['max_sharpe', 'min_volatility', 'monte_carlo'] # 'equal_wt',
        methodology = 'Option_B' #'Option_None', 'Option_A'
        # st.sidebar.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
                
                
        cols = st.sidebar.columns(4)

        with cols[0]:
            self.starter_date = str(st.sidebar.date_input(" ---> START DATE : ", datetime(2021, 7, 14)))[:10]
            self.ender_date = str(datetime.now())[:10]                    
            
        with cols[1]:
            graph1 = st.sidebar.selectbox(" ---> GRAPH :", ('Yes', 'No'), index=1)
                                                    
        with cols[2]:
              file_saver=st.sidebar.selectbox(" ---> SAVE :", ('Yes', 'No'), index=1)
        
        with cols[3]:
            if use_indicators == 'Yes':
                strategy_options = ['Single SMA', 'Double SMA', 'Bollinger Bands', 'MACD', 'RSI']
                indicator_method = st.sidebar.selectbox(label=' ---> STRATEGY :', options=strategy_options, index=1)             
                st.sidebar.markdown(f"{'__'*25}")                                                           
                

#--------------------------------------------------------------------------------------------------------------------------------------------------------
### -> -> -> METHODOLOGY & CRITERIA <- <- <- 
#--------------------------------------------------------------------------------------------------------------------------------------------------------
        

        if methodology == "Option_A":
            initial_investment = 2500.0
            ret_multiple = 0.97
            min_composite_score = 53.0
            num_portfolios = 34000
            max_allocations = 21.0
            min_sentiment_score = 11.0
            min_analyst_recom_score = 64.0
            min_rs_rating_score = 91.0
            y_factor = 0.41
            low_52wk = 1.13       

        if methodology == "Option_B":    
            num_portfolios = 34000
            initial_investment = 2500.0
            max_allocations = 19.0
            y_factor = 0.34
            low_52wk = 1.31
            ret_multiple = 0.95
            min_composite_score = 56.0
            min_sentiment_score = 13.0
            min_analyst_recom_score = 81.0
            min_rs_rating_score = 84.0     
        
        if methodology == "Option_None":
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

        if methodology == "Auto Config":
            initial_investment=2500.0
            ret_multiple=0.6
            min_composite_score = np.random.randint(50, 70)
            num_portfolios = np.random.randint(13000, 34000)
            max_allocations = np.random.randint(9, 49)
            min_sentiment_score = np.random.randint(1, 25)
            min_analyst_recom_score = np.random.randint(59, 89)
            min_rs_rating_score = np.random.randint(70, 91)
            y_factor=np.random.randint(13,51)/100      
            low_52wk=1.34


        st.sidebar.header("__[4] INITIATE__")
        if st.sidebar.button("Run Proof-Model"):
            
            self.day = str(self.starter_date)[:10]
            self.month = str(self.day)[:7]
            self.year = str(self.day)[:4]
            
            self.saveSP1500 = Path(f"data/russell_3k/{self.year}/{self.month}/{self.day}/")
            if not self.saveSP1500.exists():
                self.saveSP1500.mkdir(parents=True)               

            self.reports_portfolio = Path(f"reports/portfolio/{self.month}/{self.day}/")
            if not self.reports_portfolio.exists():
                self.reports_portfolio.mkdir(parents=True)            
                
            self.reports_portResults = Path(f"reports/port_results/{self.month}/{self.starter_date}/")
            if not self.reports_portResults.exists():
                self.reports_portResults.mkdir(parents=True)                


#--------------------------------------------------------------------------------------------------------------------------------------------------------
### -> -> -> DATA GATHER <- <- <- 
#--------------------------------------------------------------------------------------------------------------------------------------------------------   
            
            
            try:
                p999 = (f"data/russell_3k/{self.year}/{self.month}/{self.day}/russell_3k_recommender.csv")
                data = pd.read_csv(p999)
                                
            except Exception:
                st.header("No Available Data For This Day - Please Try Another Day")
                pass                                                 
                        
            data = pd.DataFrame(data)
            data.columns = [x.lower() for x in data.columns]
            data.columns = [x.replace(' ', '_') for x in data.columns]
            data.columns = [x.replace('(', '') for x in data.columns]
            data.columns = [x.replace(')', '') for x in data.columns]
            data.columns = [x.replace('/', '') for x in data.columns]
            data.columns = [x.replace('-', '') for x in data.columns]
            data.columns = [x.replace('.', '') for x in data.columns]    
            
            
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
            
            st.write(f"Total Tickers Approved In Recommender : {len(data['ticker'])}")               
            st.write(f"{'__'*25}\n{'__'*25}")
            st.subheader("random forest generated parameters: \n\n")
            st.write(f"* __initial_investment: {initial_investment}__")
            st.write(f"* __min_composite_score: {min_composite_score}__")
            st.write(f"* __num_portfolios: {num_portfolios}__")
            st.write(f"* __max_allocations: {max_allocations}__")
            st.write(f"* __min_Sentiment_Score: {min_sentiment_score}__")
            st.write(f"* __min_Analyst_Recom_score: {min_analyst_recom_score}__")
            st.write(f"* __min_rs_rating_score: {min_rs_rating_score}__")
            st.write(f"* __y_factor: {y_factor}__")
            st.write(f"{'__'*25}\n{'__'*25}")                 


            data = data.query(
                f"my_score >= {min_composite_score} \
                    and rs_rating >= {min_rs_rating_score} \
                        and sentiment_score >= {min_sentiment_score} \
                            and analyst_recom >= {min_analyst_recom_score} \
                                and returns_multiple >= {ret_multiple} \
                                    and current_price >= low_52_week * {low_52wk}"
            )            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
                            
                
            # try:
            #     del data['earnings_date']
            #     del data['ipo_date']
            # except Exception:
            #     pass

            # try:                    
            #     data = data.rename(columns={'symbol': 'ticker'})
            # except Exception:
            #     pass
            
            # try:               
            #     data = data.rename(columns={'52_week_low': 'low_52_week'})
            #     data = data.rename(columns={'52_week_high': 'high_52_week'})
            # except Exception:
            #     pass                
                
            # try:
            #     data = data.rename(columns={'relative_strength': 'rs_rating'})
            #     data = data.rename(columns={'adj_analyst_recom': 'analyst_recom'})                
            # except Exception:
            #     pass
                    
            # try:
            #     # if len(data.columns) >= 21:          
            #         for i in data.columns[7:]:
            #             data[i] = clean(data[i])      
            # except Exception:
            #     pass                    
                    
            # try:
            #     del data['unnamed:_0']
            # except Exception:
            #     pass  
            
            # try:               
            #     del data['index']
            # except Exception:
            #     pass  
            
            # try:                
            #     del data['rank']
            # except Exception:
            #     pass         
            
            
            # data['over_52wk_low'] = (data['current_price'] / data['low_52_week'])   
            # st.dataframe(data)        
            # self.recommender_dataset = pd.DataFrame(data.copy()).fillna(0.0).round(2)
            
            
            # st.write(f"Total Tickers Approved In Recommender : {len(data['ticker'])}")               
            # st.write(f"{'__'*25}\n{'__'*25}")
            # st.subheader("random forest generated parameters: \n\n")
            # st.write(f"* __initial_investment: {initial_investment}__")
            # st.write(f"* __min_composite_score: {min_composite_score}__")
            # st.write(f"* __num_portfolios: {num_portfolios}__")
            # st.write(f"* __max_allocations: {max_allocations}__")
            # st.write(f"* __min_Sentiment_Score: {min_sentiment_score}__")
            # st.write(f"* __min_Analyst_Recom_score: {min_analyst_recom_score}__")
            # st.write(f"* __min_rs_rating_score: {min_rs_rating_score}__")
            # st.write(f"* __y_factor: {y_factor}__")
            # st.write(f"{'__'*25}\n{'__'*25}")                      
            

            # data = data.query(
            #     f"my_score >= {min_composite_score} \
            #         and rs_rating >= {min_rs_rating_score} \
            #             and sentiment_score >= {min_sentiment_score} \
            #                 and analyst_recom >= {min_analyst_recom_score} \
            #                     and returns_multiple >= {ret_multiple} \
            #                         and current_price >= low_52_week * {low_52wk}"
            # )   
            
            
            
            self.recommender_dataset = data.copy().reset_index()
            port_tics = list(self.recommender_dataset["ticker"])
            
            st.write(f"Total Tickers To Model: {len(port_tics)}")
            st.dataframe(self.recommender_dataset)
                   


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
                    print(f"failed ticker {i}")
                    port_tics.remove(i)                 
            bulk_files = bulk_files.dropna(axis='columns')
            bulk_files.index = pd.to_datetime(bulk_files.index)  
            
            bulk_files = pd.DataFrame(bulk_files.reindex(sorted(bulk_files.columns), axis=1).copy())
            df_train_data = bulk_files.loc[:self.starter_date]
            df_test_data = pd.DataFrame(bulk_files.loc[self.starter_date:].copy())
            
            bulk_files.to_pickle(self.saveSP1500 / f"{methodology}.pkl")
            df_train_data.to_pickle(self.saveSP1500 / f"df_train_data.pkl")
            df_test_data.to_pickle(self.saveSP1500 / f"df_test_data.pkl") 


#--------------------------------------------------------------------------------------------------------------------------------------------------------
### -> -> -> PRINCIPAL COMPONENT ANALYSIS (PCA - FEATURIZED REDUCTION) <- <- <- 
#--------------------------------------------------------------------------------------------------------------------------------------------------------


            if graph1 == 'Yes':
                graph101 = True
            elif graph1 == 'No':
                graph101 = False
                
                
            if file_saver == 'Yes':
                return_files = True
            elif file_saver == 'No':
                return_files = False                


            if use_pca == 'Yes':  
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
                

#--------------------------------------------------------------------------------------------------------------------------------------------------------
### -> -> -> CURRENT BUY OR SELL STATUS <- <- <- 
#--------------------------------------------------------------------------------------------------------------------------------------------------------

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
                st.write(f'max_sharpe_pass = {c}/{len(port_tics)}')
                return df1


#--------------------------------------------------------------------------------------------------------------------------------------------------------
### -> -> -> MAXIMUM SHARPE PORTFOLIO REFINEMENT <- <- <- 
#--------------------------------------------------------------------------------------------------------------------------------------------------------


            if 'max_sharpe' in run_list:
                try:
                    max_sharpe_df_3 = pd.DataFrame(saver_lst_1[5]).reset_index()
                except:
                    path1 = (f"reports/portfolio/{str(self.starter_date)[:7]}/{str(self.starter_date)[:10]}/max_sharpe_df_3.pkl")
                    with open(path1, "rb" ) as fh:
                        max_sharpe_df_3 = pd.DataFrame(pickle.load(fh)).reset_index()

                try:
                    max_sharpe_df_3 = max_sharpe_df_3.rename(columns={'symbol': 'ticker'})
                except Exception:
                    pass

                if use_indicators == 'Yes':
                    max_sharpe_df = sharper(max_sharpe_df_3)                    
                else:
                    max_sharpe_df = pd.DataFrame(max_sharpe_df_3.copy())

                data = pd.DataFrame(df_test_data)
                data = data.filter(max_sharpe_df['ticker'])
                p1.Proof_of_Concept(self.starter_date, self.ender_date, return_files, True).setup(max_sharpe_df, "maximum_sharpe_ratio", data, initial_investment)
                st.write('_'*25)


#--------------------------------------------------------------------------------------------------------------------------------------------------------
### -> -> -> MINIMUM VOLATILITY PORTFOLIO <- <- <- 
#--------------------------------------------------------------------------------------------------------------------------------------------------------

            if 'min_volatility' in run_list:
                try:
                    min_vol_df_3 = pd.DataFrame(saver_lst_1[6]).reset_index()
                except:
                    path1 = (f"reports/portfolio/{str(self.starter_date)[:7]}/{str(self.starter_date)[:10]}/min_vol_df_3.pkl")
                    with open(path1, "rb" ) as fh:
                        min_vol_df_3 = pd.DataFrame(pickle.load(fh)).reset_index()

                try:
                    min_vol_df_3 = min_vol_df_3.rename(columns={'symbol': 'ticker'})
                except Exception:
                    pass

                if use_indicators == 'Yes':
                    min_vol_df = sharper(min_vol_df_3)
                else: 
                    min_vol_df = pd.DataFrame(min_vol_df_3.copy())

                data = pd.DataFrame(df_test_data)
                data = data.filter(min_vol_df['ticker'])
                p1.Proof_of_Concept(self.starter_date, self.ender_date, return_files, True).setup(min_vol_df, "minimum_volatility_portfolio", data, initial_investment)
                st.write('_'*25)                                                                                                                                                 


#--------------------------------------------------------------------------------------------------------------------------------------------------------
### -> -> -> EQUALLY WEIGHTED APPROVED MAX SHARPE RATIO COMPONENTS <- <- <- 
#--------------------------------------------------------------------------------------------------------------------------------------------------------

            if 'equal_wt' in run_list:
                path1 = (f"reports/port_results/{str(self.starter_date)[:7]}/{str(self.starter_date)[:10]}/maximum_sharpe_ratio.csv")
                equal_wt_df = pd.read_csv(path1)
                
                try:
                    equal_wt_df = equal_wt_df.rename(columns={'symbol': 'ticker'})
                except Exception:
                    pass    

                equal_wt_df["allocation"] = 100 / len(equal_wt_df["ticker"])

                data = pd.DataFrame(df_test_data)
                data = data.filter(equal_wt_df['ticker'])                        
                p1.Proof_of_Concept(self.starter_date, self.ender_date, return_files, True).setup(equal_wt_df, "maximum_sharpe_equalWT", data, initial_investment)
                st.write('_'*25)


#--------------------------------------------------------------------------------------------------------------------------------------------------------
### -> -> -> MONTE CARLO CHOLESKY PORTFOLIO <- <- <- 
#--------------------------------------------------------------------------------------------------------------------------------------------------------

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
                st.write('_'*25)


        else:
            st.subheader("NO DATA FOR THIS DAY")