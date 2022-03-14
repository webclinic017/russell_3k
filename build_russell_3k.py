import time
from datetime import datetime, timedelta
from os.path import exists
from pathlib import Path

import pandas as pd
import pickle5 as pickle
import streamlit as st
from yahoo_fin import stock_info as si
from yahooquery import Ticker

import pages as p0
import src.models.portfolio.proof as p1
import src.models.portfolio.proof_port as p2
from src.data import Source_Data
from src.gmail import The_Only_Mailer
from src.models.portfolio.proof import Proof_of_Concept
from src.models.portfolio.proof_port import The_Portfolio_Optimizer
from src.models.portfolio.web_monteCarloCholesky import \
    MonteCarloCholesky as mcc
from src.models.portfolio.web_pca import The_PCA_Analysis as pca
from src.models.recommender.simple_rec import Simple_Rick


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



class Builder(object):

    def __init__(self, day):
        self.day = day
        self.month = str(day)[:7]
        self.year = str(day)[:4]
        
        self.saveSP500 = Path(f"data/russell_3k/{self.year}/{self.month}/{self.day}/")
        if not self.saveSP500.exists():
            self.saveSP500.mkdir(parents=True)
            
        self.file_loc = Path(f"data/russell_3k/{self.year}/{self.month}/{self.day}/sp500_recommender.csv")
        if not self.file_loc.exists():
            self.file_loc.mkdir(parents=True)                

        self.reports_portfolio = Path(f"reports/portfolio/{self.month}/{self.day}/")
        if not self.reports_portfolio.exists():
            self.reports_portfolio.mkdir(parents=True)            


#----------------------------------------------------------------------------------------------------------------------------------
### -> -> -> INITIAL RECOMMENDER SYSTEM <- <- <- 
#----------------------------------------------------------------------------------------------------------------------------------            

    def build_stage_1(self):
        if not exists(self.file_loc):
            st.title('NO FILE FOUND')
            
        else:
            with open(self.file_loc, "rb") as fh:
                data = pickle.load(fh)
                df = pd.DataFrame(data).fillna(0.0)
                df.columns = [x.lower() for x in df.columns]
            try:
                del df['earnings_date']
                del df['ipo_date']
                for i in df.columns[7:]:
                    df[i] = clean(df[i])
            except Exception:
                pass

            methodology = "Defaults"
            num_portfolios = 34000
            initial_investment = 2500.0
            max_allocations = 21.0
            y_factor = 0.45
            low_52wk = 1.13
            ret_multiple = 0.97
            min_composite_score = 54.0
            min_sentiment_score = 11.0
            min_analyst_recom_score = 64.0
            min_rs_rating_score = 89.0

            # initial_investment = 2500.0
            # ret_multiple = 0.97
            # min_composite_score = 53.0
            # num_portfolios = 34000
            # max_allocations = 21.0
            # min_sentiment_score = 11.0
            # min_analyst_recom_score = 64.0
            # min_rs_rating_score = 91.0
            # y_factor = 0.41
            # low_52wk = 1.13               
            
            df['my_score'] = df['my_score'].round(2)
            df['over_52_week_low'] = (df['current_price'] / df['low_52_week'])          
            df = df[df["my_score"] >= min_composite_score]
            df = df[df["sentiment_score"] >= min_sentiment_score]
            df = df[df["analyst_recom"] >= min_analyst_recom_score]
            df = df[df["rs_rating"] >= min_rs_rating_score]            
            df = df[df["returns_multiple"] >= ret_multiple]
            df = df[df['current_price'] >= df['low_52_week'] * low_52wk]
            port_tics = list(df["ticker"])
            df = df.round(2)           
            
            st.title("__Stock Pot__")
            st.write(f"{'__'*25} \n {'__'*25}")
            st.write(f" * __Total Stocks In Pot:__ {len(df)}")
            st.dataframe(df)


        #--------------------------------------------------------------------------------------------------
        ### -> -> -> MASS DATA ACCUMULATION <- <- <- 
        #--------------------------------------------------------------------------------------------------
        

            hammerTime = Ticker(
                port_tics,
                asynchronous=True,
                backoff_factor=0.34,
                progress=True,
                validate=True,
                verify=False, 
            )
            
            hammer_hist = hammerTime.history(start="2021-01-02").reset_index().set_index('date')
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
            df_train_data = bulk_files.loc[:self.day]
            df_test_data = pd.DataFrame(bulk_files.loc[self.day:].copy())
            
            bulk_files.to_pickle(self.saveSP500 / f"{methodology}_bulk_hist.pkl")
            df_train_data.to_pickle(self.saveSP500 / f"df_train_data.pkl")            
            df_test_data.to_pickle(self.saveSP500 / f"df_test_data.pkl") 


        #------------------------------------------------------------------------------------
        ### -> -> -> PRINCIPAL COMPONENT ANALYSIS (PCA - FEATURIZED REDUCTION) <- <- <- 
        #------------------------------------------------------------------------------------


            return_files = False
            graph1 = True

            chicken_dinna = pca(
                port_tics, 
                self.day, 
                return_files, 
                y_factor
            ).build_pca(df_train_data, graph1)  

            df_pca_output = df_train_data.filter(chicken_dinna)

            saver_lst_1 = p2.The_Portfolio_Optimizer(
                self.day, 
                return_files,
            ).optimize(
                initial_investment, 
                num_portfolios, 
                max_allocations, 
                df_pca_output,
                graph1,
            )            


        #-----------------------------------------------------------------------------
        ### -> -> -> CURRENT BUY OR SELL STATUS <- <- <- 
        #-----------------------------------------------------------------------------

        crap_lst = []
        sheen_lst = []
        movAvg_strategy = 'Yes'
        strategy_style = 'Double SMA' # 'Single SMA'
        sma_ema_choice = 'SMA' # 'EMA'
        run_list = ['max_sharpe','monte_carlo',]
        self.ender_date = str(self.day)[:10]


        def sharper(df1):
            port_tics = list(df1["ticker"])
            c, cc, ccc = 0.0, 0.0, len(port_tics)

            for p in port_tics:                        
                if p in crap_lst or p in sheen_lst:
                    pass

                else:
                    temp = pd.DataFrame(hammer_hist.loc[:self.day])
                    data = pd.DataFrame(temp[temp['ticker'] == p])
                    del data['ticker']
                    del data['dividends']                      
                    try:
                        if strategy_style == 'Single SMA':
                            cc += 1
                            x = p0.Strategy(self.day).run_optimal_sma(p, data, graph1, cc, ccc)
                            if x is True:
                                c += 1
                                sheen_lst.append(p)
                            else:
                                crap_lst.append(p)
                                df1 = df1.drop(df1[df1['ticker'] == p].index)

                        if strategy_style == 'Double SMA':                                
                            cc += 1
                            x = p0.Strategy(self.day).run_movAvg_sma_ema(p, data, sma_ema_choice, graph1, cc, ccc)
                            if x is True:
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

    #-----------------------------------------------------------------------------------------------------------
    ### -> -> -> MAXIMUM SHARPE PORTFOLIO REFINEMENT <- <- <- 
    #------------------------------------------------------------------------------------------------------------


        if 'max_sharpe' in run_list:
            try:
                max_sharpe_df_3 = pd.DataFrame(saver_lst_1[5]).reset_index()
                st.dataframe(max_sharpe_df_3)
            except:
                path1 = (f"reports/portfolio/{str(self.day)[:7]}/{str(self.day)[:10]}/max_sharpe_df_3.pkl")
                with open(path1, "rb" ) as fh:
                    max_sharpe_df_3 = pd.DataFrame(pickle.load(fh)).reset_index()
            

            if movAvg_strategy == 'Yes':
                max_sharpe_df = sharper(max_sharpe_df_3)                    
            else:
                max_sharpe_df = pd.DataFrame(max_sharpe_df_3.copy())

            data = pd.DataFrame(bulk_files.loc[self.day:])
            data = data.filter(max_sharpe_df['ticker'])
            p1.Proof_of_Concept_Builder(
                self.day, self.ender_date, return_files, True).setup(max_sharpe_df, "maximum_sharpe_ratio", data, initial_investment
                )
            st.write('_'*25)            


    #-----------------------------------------------------------------------------------------------------------
    ### -> -> -> MINIMUM VOLATILITY PORTFOLIO <- <- <- 
    #-----------------------------------------------------------------------------------------------------------

        if 'min_volatility' in run_list:
            try:
                min_vol_df_3 = pd.DataFrame(saver_lst_1[6]).reset_index()
            except:
                path1 = (f"reports/portfolio/{str(self.day)[:7]}/{str(self.day)[:10]}/min_vol_df_3.pkl")
                with open(path1, "rb" ) as fh:
                    min_vol_df_3 = pd.DataFrame(pickle.load(fh)).reset_index()

            if movAvg_strategy == 'Yes':
                min_vol_df = sharper(min_vol_df_3)
            else: 
                min_vol_df = pd.DataFrame(min_vol_df_3.copy())

            data = pd.DataFrame(bulk_files.loc[self.day:])
            data = data.filter(min_vol_df['ticker'])
            p1.Proof_of_Concept_Builder(
                self.day, self.ender_date, return_files, True).setup(min_vol_df, "minimum_volatility_portfolio", data, initial_investment
                )
            st.write('_'*25)                      


    #-----------------------------------------------------------------------------------------------------------
    ### -> -> -> RANDOM CONFIGURATION PORTFOLIO <- <- <- 
    #-----------------------------------------------------------------------------------------------------------

        if 'random' in run_list:
            try:
                random_port1 = pd.DataFrame(saver_lst_1[0]).reset_index()
            except:
                path1 = (f"reports/portfolio/{str(self.day)[:7]}/{str(self.day)[:10]}/max_sharpe_df_1.pkl")
                with open(path1, "rb" ) as fh:
                    random_port1 = pd.DataFrame(pickle.load(fh)).reset_index()
            
            if movAvg_strategy == 'Yes':
                random_port = sharper(random_port1)
            else:
                random_port = pd.DataFrame(random_port1.copy())

            data = pd.DataFrame(bulk_files.loc[self.day:])
            data = data.filter(random_port['ticker'])                        
            p1.Proof_of_Concept_Builder(
                self.day, self.ender_date, return_files, True).setup(random_port, "markowitz_random", data, initial_investment
                )
            st.write('_'*25)                                         


    #-----------------------------------------------------------------------------------------------------------
    ### -> -> -> RANDOM MINIMUM VOLATILITY PORTFOLIO <- <- <- 
    #-----------------------------------------------------------------------------------------------------------

        if 'random_volatility' in run_list:
            try:
                min_vol_df_1 = pd.DataFrame(saver_lst_1[1]).reset_index()
            except:
                path1 = (f"reports/portfolio/{str(self.day)[:7]}/{str(self.day)[:10]}/min_vol_df_1.pkl")
                with open(path1, "rb" ) as fh:
                    min_vol_df_1 = pd.DataFrame(pickle.load(fh)).reset_index()
            
            if movAvg_strategy == 'Yes':
                min_vol_df = sharper(min_vol_df_1)
            else:
                min_vol_df = pd.DataFrame(min_vol_df_1.copy())

            data = pd.DataFrame(bulk_files.loc[self.day:])
            data = data.filter(min_vol_df['ticker'])                                                
            p1.Proof_of_Concept_Builder(
                self.day, self.ender_date, return_files, True).setup(min_vol_df, "minimum_volatility_random", data, initial_investment
                )
            st.write('_'*25)                                                                                  


    #-----------------------------------------------------------------------------------------------------------
    ### -> -> -> EQUALLY WEIGHTED APPROVED MAX SHARPE RATIO COMPONENTS <- <- <- 
    #-----------------------------------------------------------------------------------------------------------

        if 'equal_wt' in run_list:
            path1 = (f"reports/portfolio/{str(self.day)[:7]}/{str(self.day)[:10]}/max_sharpe_df_3.pkl")
            with open(path1, "rb" ) as fh:
                equal_wt_df = pickle.load(fh)  

            equal_wt_df["allocation"] = 100 / len(equal_wt_df["ticker"])
            equal_wt_df.columns = ['ticker', 'allocation']

            data = pd.DataFrame(bulk_files.loc[self.day:])
            data = data.filter(equal_wt_df['ticker'])                        
            p1.Proof_of_Concept_Builder(
                self.day, self.ender_date, return_files, True).setup(equal_wt_df, "maximum_sharpe_equalWT", data, initial_investment
                )
            st.write('_'*25)


    #-----------------------------------------------------------------------------------------------------------
    ### -> -> -> MONTE CARLO CHOLESKY PORTFOLIO <- <- <- 
    #-----------------------------------------------------------------------------------------------------------

        if 'monte_carlo' in run_list:
            try:
                monte_carlo_cholesky = pd.DataFrame(saver_lst_1[5]).reset_index()
            except:
                path1 = (f"reports/portfolio/{str(self.day)[:7]}/{str(self.day)[:10]}/max_sharpe_df_3.pkl")
                with open(path1, "rb" ) as fh:
                    monte_carlo_cholesky = pickle.load(fh)

            monte_carlo_cholesky = list(set(monte_carlo_cholesky['ticker']))
            num_portfolios=3400
            fin = mcc(self.day).montecarlo_sharpe_optimal_portfolio(monte_carlo_cholesky, num_portfolios, self.day)
            fin.columns = ["ticker", "allocation"]

            data = pd.DataFrame(bulk_files.loc[self.day:])
            data = data.filter(fin['ticker'])                             
            p1.Proof_of_Concept_Builder(
                self.day, self.ender_date, return_files, True).setup(fin.reset_index(), "monte_carlo_cholesky", data, initial_investment
                )
            st.write('_'*25)




        else:
            st.subheader("NO DATA FOR THIS DAY")
        

#--------------------------------------------------------------------------------------------------------------------------------------------------------
### -> -> -> EMAILER <- <- <- 
#--------------------------------------------------------------------------------------------------------------------------------------------------------


    def build_stage_2(self, day):
        The_Only_Mailer(day).mail_em_out()
        st.write("\n * [X] [COMPLETE] - Recommender Stage 8 [EMAILER]\n")


#--------------------------------------------------------------------------------------------------------------------------------------------------------
### -> -> -> RUNNER <- <- <- 
#--------------------------------------------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':

    def construction_authorized(stamp):
        b = Builder(stamp)
        b.build_stage_1()
        b.build_stage_2(stamp)

    stamper = str(st.text_input(label='Enter Date'))[:10]
    if st.button('stamper'):
        construction_authorized(stamper)
