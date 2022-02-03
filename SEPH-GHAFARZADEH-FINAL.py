# -*- coding: utf-8 -*-
"""
Momentum Trading Algorithm

Author: Seph Ghafarzadeh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

#
# Part 1 - Data Preparation
#

stock_prices = pd.read_csv('stock-prices.csv', 
                           parse_dates = True, 
                           index_col = 'date').pivot(columns = 'ticker', 
                                                     values = 'price').ffill()

snp_prices = pd.read_csv('snp-prices.csv',
                          parse_dates = True,
                          index_col = 'date')

risk_free = pd.read_csv('risk-free.csv',
                        parse_dates = True,
                        index_col = 'date')

"""
Imported given csv files for stock prices, snp prices, and risk free rate. 
Included DateTime index functionality and set index column as DateTime index. 
Forward filled stock_prices to correct missing prices.
"""

stock_rets = stock_prices.pct_change()
snp_rets = snp_prices.pct_change()

"""
Calculated returns of stock prices and snp prices with .pct_change() function, 
storing the values into new dataframes, stock_rets and snp_rets
"""

market_premium = snp_rets['spindx'] - risk_free['rf']
stock_premium = stock_rets.copy()

"""
Calculated market premium as the snp return minus the risk free rate, storing
the values in the dataframe, market_premium.
Prepared a new dataframe, stock_premium to store the values for stock premiums.
"""

for ticker in stock_rets:
    stock_premium.loc[:,ticker] = stock_rets.loc[:,ticker] - risk_free['rf']
    
"""
Calculated stock premium for each stock by iterating through each column of 
the dataframe, stock_rets, and subtracting the risk free rate from each value,
storing the new values in the dataframe, stock_premium.
"""
    
form = {}
for window in [5, 30, 60, 90, 120]:
     form[window] = stock_premium.rolling(window).apply(
         lambda r: (1+r).prod() - 1, raw = True)
 
"""
Iterated through window sizes, using the window size in the current
iteration as the rolling window to calculate formation period returns, storing
the values in a dictionary of dataframes, form.
Call different formation period dataframes with form[window]
"""

hpr = {}
for window in [5, 60, 120]:
    hpr[window] = form[window].shift(-window)
    
"""
Iterated through window sizes, using the function .shift(-window size) 
to shift back the calculated formation periods by the current iteration of the
window size, effectively converting the backwards viewing formation period returns
into forwards viewing holding period returns, and storing the values in the
dictionary of dataframes, hpr.
Call different hpr dataframes with hpr[window].
"""

#
# Part 2 - Algorithm Execution 
#

f5_ranks = form[5].rank(axis=1, ascending = False)
f5_max = form[5].rank(axis=1, ascending = False).max(axis=1)
f30_ranks = form[30].rank(axis=1, ascending = False)
f30_max = form[30].rank(axis=1, ascending = False).max(axis=1)
f90_ranks = form[90].rank(axis=1, ascending = False)
f90_max = form[90].rank(axis=1, ascending = False).max(axis=1)

"""
Ranked all values by row in formation period dataframes in descending
order with the function .rank() and stored these values. Also found the maximum
rank within each row of each formation period dataframe, and stored this value
to also be used for the algorithm below:
    
The following algorithm was used for the trading strategies below: 
I first called the top 5 ranks within each formation period (all ranks <= 5) 
and the lowest 5 ranks within each formation period (all ranks >= the minimum rank + 5) 
and then used these parameters to call the same location of values in the appropriate
hpr dataframe. A positive weighted average of equal weights was calculated for
the hpr values associated with the top 5 formation period ranks, and a negative
weighted average of equal weights was calculated for the hpr values associated with
the lowest 5 ranks. These weighted averages were then summated to find the daily
return for the combination of a given formation period and hpr, and these values
were stored in a dataframe named by its according strategy. 
Additionally, these dataframes with daily return values was used to calculate 
cumulative returns for each strategy and was stored in a separate dataframe with
the appropriate label. 
Ultimately, both the daily returns dataframe and the cumulative returns dataframe were
combined into a single dataframe with the pd.concat function, and these dataframes
were exported as csv files as instructed. 
"""

f5hpr5 = hpr[5][
    f5_ranks <= 5].apply(lambda r: (0.2 * r).sum(), axis = 1) + hpr[5][
        f5_ranks >= f5_ranks.eq((f5_max - 5), axis = 0)].apply(lambda r: (-0.2 * r).sum(), axis = 1)
f5hpr5_cumrets = (1 + f5hpr5).cumprod() - 1  
portfolio_f5hpr5 = pd.concat([f5hpr5, f5hpr5_cumrets], axis = 1).rename(
    columns = {0: "rets", 1: "cum rets"}).to_csv('portfolio-f5hpr5.csv')

""""
Portfolio 1: 5 day formation period, 5 day holding period returns
"""

f5hpr60 = hpr[60][
    f5_ranks <= 5].apply(lambda r: (0.2 * r).sum(), axis = 1) + hpr[60][
        f5_ranks >= f5_ranks.eq((f5_max - 5), axis = 0)].apply(lambda r: (-0.2 * r).sum(), axis = 1)  
f5hpr60_cumrets = (1 + f5hpr60).cumprod() - 1            
portfolio_f5hpr60 = pd.concat([f5hpr60, f5hpr60_cumrets], axis = 1).rename(
    columns = {0: "rets", 1: "cum rets"}).to_csv('portfolio-f5hpr60.csv')

""""
Portfolio 2: 5 day formation period, 60 day holding period returns
"""
         
f5hpr120 = hpr[120][
    f5_ranks <= 5].apply(lambda r: (0.2 * r).sum(), axis = 1) + hpr[120][
        f5_ranks >= f5_ranks.eq((f5_max - 5), axis = 0)].apply(lambda r: (-0.2 * r).sum(), axis = 1)
f5hpr120_cumrets = (1 + f5hpr120).cumprod() - 1
portfolio_f5hpr120 = pd.concat([f5hpr120, f5hpr120_cumrets], axis = 1).rename(
    columns = {0: "rets", 1: "cum rets"}).to_csv('portfolio-f5hpr120.csv')

""""
Portfolio 3: 5 day formation period, 120 day holding period returns
"""
        
f30hpr5 = hpr[5][
    f30_ranks <= 5].apply(lambda r: (0.2 * r).sum(), axis = 1) + hpr[5][
        f30_ranks >= f30_ranks.eq((f30_max - 5), axis = 0)].apply(lambda r: (-0.2 * r).sum(), axis = 1)
f30hpr5_cumrets = (1 + f30hpr5).cumprod() - 1
portfolio_f30hpr5 = pd.concat([f30hpr5, f30hpr5_cumrets], axis = 1).rename(
    columns = {0: "rets", 1: "cum rets"}).to_csv('portfolio-f30hpr5.csv')

""""
Portfolio 4: 30 day formation period, 5 day holding period returns
"""
        
f30hpr60 = hpr[60][
    f30_ranks <= 5].apply(lambda r: (0.2 * r).sum(), axis = 1) + hpr[60][
        f30_ranks >= f30_ranks.eq((f30_max - 5), axis = 0)].apply(lambda r: (-0.2 * r).sum(), axis = 1)
f30hpr60_cumrets = (1 + f30hpr60).cumprod() - 1
portfolio_f30hpr60 = pd.concat([f30hpr60, f30hpr60_cumrets], axis = 1).rename(
    columns = {0: "rets", 1: "cum rets"}).to_csv('portfolio-f30hpr60.csv')

""""
Portfolio 5: 30 day formation period, 60 day holding period returns
"""
             
f30hpr120 = hpr[120][
    f30_ranks <= 5].apply(lambda r: (0.2 * r).sum(), axis = 1) + hpr[120][
        f30_ranks >= f30_ranks.eq((f30_max - 5), axis = 0)].apply(lambda r: (-0.2 * r).sum(), axis = 1)
f30hpr120_cumrets = (1 + f30hpr120).cumprod() - 1
portfolio_f30hpr120 = pd.concat([f30hpr120, f30hpr120_cumrets], axis = 1).rename(
    columns = {0: "rets", 1: "cum rets"}).to_csv('portfolio-f30hpr120.csv')

""""
Portfolio 6: 30 day formation period, 120 day holding period returns
"""
        
f90hpr5 = hpr[5][
    f90_ranks <= 5].apply(lambda r: (0.2 * r).sum(), axis = 1) + hpr[5][
        f90_ranks >= f90_ranks.eq((f90_max - 5), axis = 0)].apply(lambda r: (-0.2 * r).sum(), axis = 1)
f90hpr5_cumrets = (1 + f90hpr5).cumprod() - 1
portfolio_f90hpr5 = pd.concat([f90hpr5, f90hpr5_cumrets], axis = 1).rename(
    columns = {0: "rets", 1: "cum rets"}).to_csv('portfolio-f90hpr5.csv')

""""
Portfolio 7: 90 day formation period, 5 day holding period returns
"""
        
f90hpr60 = hpr[60][
    f90_ranks <= 5].apply(lambda r: (0.2 * r).sum(), axis = 1) + hpr[60][
        f90_ranks >= f90_ranks.eq((f90_max - 5), axis = 0)].apply(lambda r: (-0.2 * r).sum(), axis = 1)   
f90hpr60_cumrets = (1 + f90hpr60).cumprod() - 1
portfolio_f90hpr60 = pd.concat([f90hpr60, f90hpr60_cumrets], axis = 1).rename(
    columns = {0: "rets", 1: "cum rets"}).to_csv('portfolio-f90hpr60.csv')

""""
Portfolio 8: 90 day formation period, 60 day holding period returns
"""
        
f90hpr120 = hpr[120][
    f90_ranks <= 5].apply(lambda r: (0.2 * r).sum(), axis = 1) + hpr[120][
        f90_ranks >= f90_ranks.eq((f90_max - 5), axis = 0)].apply(lambda r: (-0.2 * r).sum(), axis = 1)
f90hpr120_cumrets = (1 + f90hpr120).cumprod() - 1
portfolio_f90hpr120 = pd.concat([f90hpr120, f90hpr120_cumrets], axis = 1).rename(
    columns = {0: "rets", 1: "cum rets"}).to_csv('portfolio-f90hpr120.csv')  

""""
Portfolio 9: 90 day formation period, 120 day holding period returns
"""
 
#
# Part 3 - Analysis
#

avg_rets = []
min_rets = []
max_rets = []

for portfolio in [f5hpr5, f5hpr60, f5hpr120,
                  f30hpr5, f30hpr60, f30hpr120,
                  f90hpr5, f90hpr60, f90hpr120,]:
    avg_rets.append(np.average(portfolio))
    min_rets.append(np.min(portfolio))
    max_rets.append(np.max(portfolio))
    plt.plot(portfolio)
    
plt.xlabel('Date')
plt.ylabel('Returns (%)')
plt.ylim(-3.5, 1.8)
plt.title(label='Daily Returns of Portfolios')

"""
Iterated through the dataframes for each strategy, calculating the average
daily return, minimum daily return, maximum daily return, and plotting
the values of each dataframe (the daily returns for each strategy) with 
the date as the X axis and the returns as the Y axis. 
"""
        
d = {'port avg rets' : avg_rets,
     'port min rets' : min_rets,
     'port max rets' : max_rets,
     'snp avg rets' : [market_premium.mean()]*len(avg_rets)}

port_perform = pd.DataFrame(d,
                               index = ['Form5HPR5',
                                        'Form5HPR60',
                                        'Form5HPR120',
                                        'Form30HPR5',
                                        'Form30HPR60',
                                        'Form30HPR120',
                                        'Form90HPR5',
                                        'Form90HPR60',
                                        'Form90HPR120'])

"""
Created a new dataframe, port_perform, to store the performance of each 
strategy. Each strategy's average daily reutrn, minimum daily return, maximum daily return
was stored in the df's columns, with the description of each strategy as the index,
and an additional column for the market premium avg return was made as well (snp avg rets).
"""

port_perform['port diff'] = port_perform['port avg rets'] - port_perform['snp avg rets']

"""
Calculated the difference between the average daily return for each strategy with the
average daily return of the market premium.
"""

print(port_perform)
print(f'Best Performer: {port_perform["port avg rets"].max()}')










