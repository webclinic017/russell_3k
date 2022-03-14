# the_only_game_in_town
the_only_game_in_town

# An Advisor App Designed To Replace Your Financial Advisor:
------------------------------------------------------------------------------------------------------------------------
 * Screening the entire stock market
 * Scraping registered financial analysts' recommendations
 * Applying an array of fundamental & technical analysis techniques 
 * Ensemble Layewring supervised & unsupervised machine learning models to refine lean investment grade tickers
 * Returning a polished & managable short list of Investable securities
 * Visualize multiple machine learning forecasting tools
 * Construct the ideal portfolio with multiple options on weights and positions from the short list
 * Identify the most important feature stocks within the short list
 * Visualize multiple trading strategies for the refined advisor stock list
 * Pin-point positions to act on with the strategy tool generating exact entry & exit points
 * Outperform the market, your buddies, and your old financial advisor


# Website URL: [The Only Game In Town](https://share.streamlit.io/ramraider011235/the_only_game_in_town/main/app.py/)
------------------------------------------------------------------------------------------------------------------------


# Project Organization:
------------------------------------------------------------------------------------------------------------------------
    ╠══ .streamlit
    ║     ╠══ config.toml
    ║     ╚══ secrets.toml
    ║
    ╠══ .vscode
    ║     ╚══ settings.json
    ║
    ╠══ data
    ║     ╠══ advisor    
    ║     ╠══ bin
    ║     ╠══ bunker
    ║     ╠══ finviz
    ║     ╠══ images
    ║     ╠══ junk             
    ║     ╠══ plots
    ║     ╠══ proof
    ║     ╠══ raw
    ║     ╠══ recommenders
    ║     ╠══ screeners
    ║     ╠══ sentiment
    ║     ╠══ tickers
    ║     ╚══ variates      
    ║
    ╠══ pages
    ║     ╠══ __init__.py    
    ║     ╠══ analysis.py    
    ║     ╠══ backtest.py
    ║     ╠══ forecast.py    
    ║     ╠══ home.py
    ║     ╠══ login.py    
    ║     ╠══ portfolio.py
    ║     ╠══ proof_pg.py    
    ║     ╠══ recommender.py
    ║     ╠══ screeners.py
    ║     ╠══ snapshot.py    
    ║     ╚══ strategy.py    
    ║
    ╠══ reports
    ║     ╠══ measurements   
    ║     ║     ╚══ dick_measurement.csv
    ║     ║         
    ║     ╠══ port_results
    ║     ║     ╠══ 2021-07
    ║     ║     ╠══ 2021-08
    ║     ║     ╠══ 2021-09
    ║     ║     ╠══ 2021-10
    ║     ║     ╠══ 2021-11
    ║     ║     ╠══ 2021-12
    ║     ║     ╠══ 2022-01
    ║     ║     ╚══ 2022-02
    ║     ║
    ║     ╠══ portfolio
    ║     ║     ╠══ 2021-07
    ║     ║     ╠══ 2021-08
    ║     ║     ╠══ 2021-09
    ║     ║     ╠══ 2021-10
    ║     ║     ╠══ 2021-11
    ║     ║     ╠══ 2021-12
    ║     ║     ╠══ 2022-01
    ║     ║     ╚══ 2022-02
    ║     ║        
    ║     ╚══ score_sheet
    ║           ╠══ mondays.csv
    ║           ╠══ tuesdays.csv
    ║           ╠══ wednesdays.csv
    ║           ╠══ thursdays.csv
    ║           ╠══ fridays.csv
    ║           ╠══ random_0.csv
    ║           ╠══ random_choice.csv
    ║           ╚══ printin_press.csv
    ║          
    ║
    ╠══ src
    ║     ╠══ data
    ║     ║     ╠══ __init__.py
    ║     ║     ╠══ source_data.py
    ║     ║     ╚══ yahoo_fin_stock_info.py
    ║     ║
    ║     ╠══ gmail
    ║     ║     ╠══ __init__.py
    ║     ║     ╠══ gmail.py
    ║     ║     ╠══ tutorial.ipynb    
    ║     ║     ╚══ credentials.json 
    ║     ║
    ║     ╠══ models
    ║     ║     ╚══ analysis
    ║     ║     ║     ╠══ __init__.py    
    ║     ║     ║     ╠══ CAPM_CAGR.py    
    ║     ║     ║     ╠══ financial_signal_processing.py
    ║     ║     ║     ╠══ multivariate_timeSeries_rnn.py
    ║     ║     ║     ╚══ single_asset_analysis.py
    ║     ║     ║  
    ║     ║     ╚══ backtest
    ║     ║     ║     ╠══ __init__.py                      
    ║     ║     ║     ╠══ A1.py
    ║     ║     ║     ╠══ B1.py
    ║     ║     ║     ╠══ optimal_double_mavg.py
    ║     ║     ║     ╠══ optimal_sma.py
    ║     ║     ║     ╠══ vectorized_backtest.py
    ║     ║     ║     ╠══ web_backtrader_sma_strategy.py                
    ║     ║     ║     ╚══ web_one.py
    ║     ║     ║    
    ║     ║     ╚══ forecast
    ║     ║     ║     ╠══ __init__.py
    ║     ║     ║     ╠══ web_arima.py
    ║     ║     ║     ╠══ web_mc.py        
    ║     ║     ║     ╠══ web_monteCarlo.py
    ║     ║     ║     ╠══ web_prophet.py
    ║     ║     ║     ╠══ web_regression.py
    ║     ║     ║     ╠══ web_sarima.py
    ║     ║     ║     ╠══ web_stocker_helper.py
    ║     ║     ║     ╠══ web_stocker.py
    ║     ║     ║     ╠══ web_univariate_rnn.py
    ║     ║     ║     ╚══ web_univariate_timeSeries_rnn.p
    ║     ║     ║
    ║     ║     ╚══ portfolio
    ║     ║     ║     ╠══ __init__.py
    ║     ║     ║     ╠══ portfolio_analysis.py
    ║     ║     ║     ╠══ proof.py
    ║     ║     ║     ╠══ proof_port.py
    ║     ║     ║     ╠══ prove_portfolio.py                 
    ║     ║     ║     ╠══ web_efficient_frontier.py
    ║     ║     ║     ╠══ web_pca.py
    ║     ║     ║     ╠══ web_plot_roc.py
    ║     ║     ║     ╠══ web_portfolio_optimizer.py 
    ║     ║     ║     ╚══ web_random_forest.py
    ║     ║     ║
    ║     ║     ╚══ recommender
    ║     ║           ╠══ __init__.py
    ║     ║           ╚══ recommender_composite.py          
    ║     ║           ║
    ║     ║           ╚══ strategy
    ║     ║           ║     ╠══ __init__.py
    ║     ║           ║     ╠══ web_movingAverage_sma_ema.py
    ║     ║           ║     ╠══ web_optimal_double_sma.py
    ║     ║           ║     ╠══ web_optimal_sma.py
    ║     ║           ║     ╠══ web_overBought_overSold.py
    ║     ║           ║     ╠══ web_support_resistance.py
    ║     ║           ║     ╚══ web_trading_technicals.py
    ║     ║           ║
    ║     ║           ╚══ web_construction
    ║     ║                 ╚══ dash_projects
    ║     ║                 ╚══ st_files
    ║     ║      
    ║     ╚══ tools
    ║           ╠══ __init__.py    
    ║           ╠══ functions.py
    ║           ╠══ lists.py
    ║           ╠══ scripts.py
    ║           ╚══ widgets.py
    ║
    ╠══ .gitignore
    ╠══ app.py
    ╠══ build.py
    ╠══ LICENSE
    ╠══ README.md
    ╚══ requirements.txt
                                                
                                                

------------------------------------------------------------------------------------------------------------------------