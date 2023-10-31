import streamlit as st

import pandas as pd
import numpy as np
from tvDatafeed import TvDatafeed, Interval

# HAVL Libraries
from scipy.signal import argrelextrema
from collections import deque

# Bokeh things
from bokeh.plotting import figure, output_file, show
from bokeh.models import Range1d
from bokeh.io import output_notebook

# ~~~~~ METHODS ~~~~~ #
def get_market_data(name, pair, timeframe, market):
    asset = name + pair # e.g. 'vetusd'
    symbs = tv.search_symbol(asset) # not case sensitive
    
    if timeframe == '1d':
        tf_interval = Interval.in_daily
    elif timeframe == '4h':
        tf_interval = Interval.in_4_hour
    elif timeframe == '5m':
        tf_interval = Interval.in_5_minute
    
    df = list()
    for el in symbs:
        el_symbol = el['symbol']
        el_market = el['exchange']
        if asset.upper() == el_symbol: # .upper() makes 'vetusd' -> 'VETUSD', needed because case sensitive
            if market == el_market:
                data = tv.get_hist(
                    symbol = el_symbol,
                    exchange = el_market,
                    interval = tf_interval,
                    n_bars = 10000
                )
                df = data.copy()

    return df


def getHigherHighs(data: np.array, order, K, tau = 0):
    
    # get indices and values of all highs of given order
    high_idx = argrelextrema(data, np.greater, order=order)[0]
    highs = data[high_idx]
    
    extrema = []
    extr = 0
    ex_deque = deque()
    #last_high_idx = 0
    
    # ensure consecutive highs are higher than previous highs
    for i, idx in enumerate(high_idx):
        #print(i,idx,highs[i],'\tdelta: ', idx - extr)
        
        # implementing constant decorrelation time
        if idx - extr > tau:
            if i == 0:
                ex_deque.append(idx)
                continue

            #higher highs, condition is <
            if highs[i] < highs[i-1]:
                ex_deque.clear()
                ex_deque.append(idx)
            else:
                ex_deque.append(idx)

            if len(ex_deque) == K:
                extrema.append(ex_deque.copy())
                ex_deque.clear()
                extr = extrema[-1][1]
        else:
            ex_deque.clear()
        
    return extrema





# ~~~~~ MAIN ~~~~~ #

tv = TvDatafeed()

st.write("# Higher Highs Crypto!")
st.write("This is a basic app for visualizing crypto related data using trading view datafeed. Furthermore the app is willing to show the last higher high pattern for the desired cryptocurrency in order to check what happens fter the pattern of a Higher High occured!")

name = st.text_input("Insert crypto (e.g. doge): ", "doge")
pair = st.text_input("Insert pair exchange (or dominance if crypto is btc), e.g. (usd, usdt, .D): ", "usd")
market = st.text_input("Choose market:", "BINANCE")
timeframe = st.text_input("Choose Timeframe: (e.g. 1d, 4h, 5m)", "4d")

data = get_market_data(name, pair, timeframe, market)

# Plot the big timeseries
inc = data.close > data.open
dec = data.open > data.close

TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

p = figure(x_axis_type="datetime", tools=TOOLS, width=2000, title = "OHLC Chart", y_axis_type = 'log')
p.grid.grid_line_alpha=0.9
p.segment(data.index, data.high, data.index, data.low, color="black")
p.vbar(data.index[inc], w, data.open[inc], data.close[inc], fill_color="#90ee90", line_color="black")
p.vbar(data.index[dec], w, data.open[dec], data.close[dec], fill_color="#f08080", line_color="black")

st.write(data)
st.bokeh_chart(p, use_container_width=True)




st.write(f"# Explore Higher High possible patterns!")

# PARAMETERS for SAMPLES
lf = 3 # look forward for establishing targets
order = 4 # order of the relative max
tau = 12 # decorrelation time in # of candles, on time series of 4h it's 48h

K = 2 # consecutive highs
#period = 14 # period of indicators

lf = st.text_input("Insert lookforward parameter (e.g. x1, x2, x3): ", "1")
order = st.text_input("Insert max order (e.g. 2, 3, 4): ", "3")
tau = st.text_input("Insert temporal distance between samples (e.g. 12 candles)", "12")

hh = getHigherHighs(np.array(df.close), order, K, tau)
hh_idx_list = [ list(samp) for samp in hh ]
hh_idx_array = np.array(hh_idx_list)
hh_samp = list()
for hh_idx in hh_idx_list:
    d = hh_idx[-1] - hh_idx[0] # this is # of candles from 1st max to the 2nd
    s = hh_idx[0] - order # this is the start of the pattern
    e = hh_idx[-1] + int(lf*(d)) + order # this is the end of the pattern
    
    hh_samp.append([s,e])

df_hh = list()
for samp_idx in hh_samp:
    df_hh.append(df.iloc[samp_idx[0] : samp_idx[1]].copy()) # take df values from the start till the end

st.write(f"Number of Higher High Samples: {len(df_hh)}")
idx = st.text_input("Insert a index for the sample to visualize (e.g. 42, 89...): ", f"{len(df_hh)-1}")

df_i_sample = df_hh[idx].copy()
# convert datetime to timestamp
df_i_sample['datetime'] = df_i_sample.index
df_i_sample['timestamp'] = df_i_sample.index.to_series().apply(lambda x: x.timestamp()*1000) # to milliseconds
df_i_sample.set_index('timestamp', inplace = True)

price = np.array(df_i_sample.close)
dates = df_i_sample.index

hh = getHigherHighs(price, order, K)
hh_idx = list(hh[0]) # convert deque to list

# setting coordinates for the segment of the pattern
x_hh = hh_idx
y_hh = df_i_sample.close.iloc[hh_idx].to_list()

x_hh_conf = [ i+order for i in hh_idx ]
y_hh_conf = df_i_sample.close.iloc[x_hh_conf].to_list()

# Mapping of positions in dataframe to correct timestamp required because we set x_axis_type = datetime
x_hh_map = [df_i_sample.index[x_hh[0]], df_i_sample.index[x_hh[-1]]]
x_hh_conf_map = [df_i_sample.index[x_hh_conf[0]], df_i_sample.index[x_hh_conf[-1]]]
candle_milliseconds = 14400*1000 # 4h

TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

sp = figure(
    title = f"Higher High Sample - {df_i_sample.datetime.iloc[0]} to {df_i_sample.datetime.iloc[-1]}",
    x_axis_type="datetime",
    tools=TOOLS,
    plot_width=700)

sp.grid.grid_line_alpha = 1 # intensity of grid appearance

inc = df_i_sample['close'] > df_i_sample['open']
dec = df_i_sample['open'] > df_i_sample['close']
w = candle_milliseconds*0.8 # width

# Plotting candlesticks
sp.segment(df_i_sample.index, df_i_sample['high'], df_i_sample.index, df_i_sample['low'], color="black")
sp.vbar(df_i_sample.index[inc], w, df_i_sample.open[inc], df_i_sample.close[inc], fill_color="#00af50", line_color="black")
sp.vbar(df_i_sample.index[dec], w, df_i_sample.open[dec], df_i_sample.close[dec], fill_color="#F2583E", line_color="black")

# Plotting pattern and confirmation markers
sp.line(x_hh_map, y_hh, color='red', width = 4)
sp.scatter(x_hh_conf_map, y_hh_conf, marker = "circle", size = 10, color = 'black',fill_color = "black")

# retracement classes
width = candle_milliseconds*( x_hh[-1] - x_hh[0] )*lf
height_soft = 0.05*y_hh[-1]
height_norm = 0.15*y_hh[-1] - height_soft
height_hard = y_hh[-1] - height_norm - height_soft

sp.rect(x = x_hh_map[-1] + int(width/2.), y = y_hh[-1] - height_soft/2., #setting center
       width = width, height = height_soft,
       color='green', fill_color = "green", alpha = 0.1)
sp.rect(x = x_hh_map[-1] + int(width/2.), y = y_hh[-1] - height_soft - height_norm/2,
       width = width, height = height_norm,
       color='yellow', fill_color = "yellow", alpha = 0.1)
sp.rect(x = x_hh_map[-1] + int(width/2.), y = y_hh[-1] - height_soft - height_norm - height_hard/2,
       width = width, height = height_hard,
       color='orange', fill_color = "orange", alpha = 0.1)

sp.y_range = Range1d(min(df_i_sample['low']), max(df_i_sample['high']))
st.bokeh_chart(sp, use_container_width=True)