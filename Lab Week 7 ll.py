# -*- coding: utf-8 -*-
"""
Created on Thu May  2 19:42:14 2024

@author: ChelseySSS
"""

import pandas as pd
import pandas_datareader.data as web
import statsmodels.api as sm
import matplotlib.pyplot as plt

series = {'CPMNACSCAB1GQDE':'GDPGermany',
          'LRUNTTTTDEQ156S':'EMPGermany',
          'CPMNACSCAB1GQPL':'GDPPoland',
          'LRUNTTTTPLQ156S':'EMPPoland'}
df = web.DataReader(series.keys(), 'fred', start='1995-01-01', end='2019-10-01')

df = df.rename(series, axis=1)

# 1)
# This data is from lecture 18.  Explore it using plots and summary
# statistics. What is wrong with the employment data from Poland? 
# Then, apply an HP filter from the statsmodels library, and filter 
# all four series.  Plot the cycles, trends, and original values to
# see what is happening when you filter.

# 1) Explore the data using plots and summary statistics
print(df.describe())

# Plotting each series to identify potential issues
fig, axs = plt.subplots(4, 1, figsize=(10, 15))
for i, col in enumerate(df.columns):
    axs[i].plot(df.index, df[col])
    axs[i].set_title(col)
    axs[i].set_xlabel('Year')
    axs[i].set_ylabel(col)
plt.tight_layout()
plt.show()

# 2) Identify potential issues in EMPPoland data
# It's common to check for unusual values or trends that don't make sense.
# We plot it first to visually inspect.
plt.figure(figsize=(10, 5))
plt.plot(df['EMPPoland'], label='EMPPoland')
plt.title('Employment Data for Poland')
plt.xlabel('Year')
plt.ylabel('Employment Rate')
plt.legend()
plt.show()

# 3) Applying HP filter and plotting results
from statsmodels.tsa.filters.hp_filter import hpfilter

# HP filter application
cycle, trend = {}, {}
for col in df.columns:
    cycle[col], trend[col] = hpfilter(df[col], lamb=1600)

# Plotting cycles, trends, and original values
fig, axs = plt.subplots(4, 3, figsize=(15, 20))
for i, col in enumerate(df.columns):
    axs[i, 0].plot(df.index, df[col], label='Original')
    axs[i, 0].set_title(f'Original {col}')
    axs[i, 1].plot(df.index, trend[col], label='Trend', color='red')
    axs[i, 1].set_title(f'Trend {col}')
    axs[i, 2].plot(df.index, cycle[col], label='Cycle', color='green')
    axs[i, 2].set_title(f'Cycle {col}')
    for j in range(3):
        axs[i, j].set_xlabel('Year')
        axs[i, j].set_ylabel(col)
        axs[i, j].legend()
plt.tight_layout()
plt.show()


# 2)
# The code from the lecture includes a function that implements the
# Hamilton filter, though we did not go over the code in detail.
# Copy that function over and try to understand most of what it is
# doing (you may have to test it in pieces) and then apply it to
# this data. Modify your plots from question 1 to compare the results
# of the Hamilton and HP filters to the unfiltered values.

# Assuming Hamilton filter function 'hamilton_filter' is provided and implemented
# Example function signature might be: def hamilton_filter(series, window_size):

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from statsmodels.tsa.filters.hp_filter import hpfilter

# Hamilton filter implementation
def hamilton_filter(y, h=8):
    T = len(y)
    y_pad = np.pad(y, (h, h), 'edge')
    trend = np.zeros(T)
    for t in range(T):
        window = y_pad[t:t+2*h+1]
        trend[t] = np.median(window)
    return trend

# Load the data
series = {'CPMNACSCAB1GQDE': 'GDPGermany',
          'LRUNTTTTDEQ156S': 'EMPGermany',
          'CPMNACSCAB1GQPL': 'GDPPoland',
          'LRUNTTTTPLQ156S': 'EMPPoland'}
df = web.DataReader(series.keys(), 'fred', start='1995-01-01', end='2019-10-01')
df = df.rename(series, axis=1)

# Applying the HP and Hamilton filters
hp_trends, hamilton_trends = {}, {}
for col in df.columns:
    cycle, trend = hpfilter(df[col], lamb=1600)
    hp_trends[col] = trend
    hamilton_trends[col] = hamilton_filter(df[col])

# Plotting the results
fig, axs = plt.subplots(4, 3, figsize=(15, 20))
for i, col in enumerate(df.columns):
    axs[i, 0].plot(df.index, df[col], label='Original', color='blue')
    axs[i, 0].set_title(f'Original {col}')
    axs[i, 1].plot(df.index, hp_trends[col], label='HP Trend', color='red')
    axs[i, 1].set_title(f'HP Trend {col}')
    axs[i, 2].plot(df.index, hamilton_trends[col], label='Hamilton Trend', color='green')
    axs[i, 2].set_title(f'Hamilton Trend {col}')
    for j in range(3):
        axs[i, j].set_xlabel('Year')
        axs[i, j].set_ylabel(col)
        axs[i, j].legend()
plt.tight_layout()
plt.show()
