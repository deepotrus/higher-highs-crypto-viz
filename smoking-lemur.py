import streamlit as st

import pandas as pd
import numpy as np
from tvDatafeed import TvDatafeed, Interval

# HAVL Libraries
from scipy.signal import argrelextrema
from collections import deque

from bokeh.plotting import figure, output_file, show
from bokeh.models import Range1d # set y range axis
from bokeh.models.annotations import Label
from bokeh.core.properties import value # for plotting images from urls
from bokeh.io import output_notebook
from bokeh.io import export_png
from bokeh.io import curdoc # set bokeh theme

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

##################################################
# ~~~~~~~~~~~~~~~~~ GENERAL METHODS ~~~~~~~~~~~~~~~~~~~~~#
##################################################

def get_market_data(name, pair, timeframe, market):
    asset = name + pair # e.g. 'vetusd'
    symbs = tv.search_symbol(asset) # not case sensitive
    
    if timeframe == '1d':
        tf_interval = Interval.in_daily
        w = 24 * 60 * 60 * 1000 # 12 hours in milliseconds, usually half of timeframe
        #   h    min   s    ms
    elif timeframe == '4h':
        tf_interval = Interval.in_4_hour
        w = 4 * 60 * 60 * 1000
    elif timeframe == '5m':
        tf_interval = Interval.in_5_minute
        w = 1 * 5 * 60 * 1000
    
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

    return df, w

##################################################
# ~~~~~~~~~~~~~~~~~ HAVL METHODS ~~~~~~~~~~~~~~~~#
##################################################

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


def get_target_class(price, x_hh, y_hh, order):
    pctv = ( price[x_hh[-1]:].min() - y_hh[-1] ) / y_hh[-1]

    if pctv < -0.15:
        target = "crash"
    else:
        if price[-order] >= y_hh[-1]:
            target = "bull"
        else:
            target = "bear"
    return target

def get_X_y(Xy):
    y = np.array(Xy['target'])
    Xy = Xy.drop('target', axis = 1)
    X = np.array(Xy)
    
    return X, y

##################################################
# ~~~~~~~~~~~ PLOTTING METHODS ~~~~~~~~~~~~~~~~~~#
##################################################

def plot_big_timeseries(data, w, name, pair):
    # Plot the big timeseries
    inc = data.close > data.open
    dec = data.open > data.close
    
    TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

    p = figure(x_axis_type="datetime", tools=TOOLS, width=2000, title = f"{name.upper()}{pair.upper()}", y_axis_type = 'log')
    p.grid.grid_line_alpha=0.9
    p.segment(data.index, data.high, data.index, data.low, color="black")
    p.vbar(data.index[inc], w, data.open[inc], data.close[inc], fill_color="#90ee90", line_color="black")
    p.vbar(data.index[dec], w, data.open[dec], data.close[dec], fill_color="#f08080", line_color="black")
    
    return p

def prep_plot(dfi):
    price = np.array(dfi.close)
    dates = dfi.index
    candle_milliseconds = int((dates[1]-dates[0])/1000)
    
    hh = getHigherHighs(price, order, K)
    hh_idx = list(hh[0]) # convert deque to list
    
    # setting coordinates for the segment of the pattern
    x_hh = hh_idx
    y_hh = dfi.close.iloc[x_hh].to_list()
    
    length = x_hh[-1] - x_hh[0]
    
    x_hh_conf = [ i+order for i in hh_idx ]
    y_hh_conf = dfi.close.iloc[x_hh_conf].to_list()

    # Mapping of positions in dataframe to correct timestamp required because we set x_axis_type = datetime
    x_hh_map = [dfi.index[x_hh[0]], dfi.index[x_hh[-1]]]
    x_hh_conf_map = [dfi.index[x_hh_conf[0]], dfi.index[x_hh_conf[-1]]]
    
    return x_hh_map, x_hh_conf_map, length, y_hh, y_hh_conf, candle_milliseconds

def candles_plot(p, dfi, width_candles):
    inc = dfi['close'] > dfi['open']
    dec = dfi['open'] > dfi['close']

    # Plotting candlesticks
    p.segment(dfi.index, dfi['high'], dfi.index, dfi['low'], color="black")
    p.vbar(dfi.index[inc], width_candles, dfi.open[inc], dfi.close[inc], fill_color="#00af50", line_color="black")
    p.vbar(dfi.index[dec], width_candles, dfi.open[dec], dfi.close[dec], fill_color="#F2583E", line_color="black")
    
    return p

def objects_plot(p, x_hh_map, x_hh_conf_map, y_hh, y_hh_conf):
    # Plotting pattern and confirmation markers
    p.line(x_hh_map, y_hh, color='purple', width = 4, legend_label="Higher High Pattern")
    p.scatter(x_hh_conf_map, y_hh_conf, marker = "circle", size = 10, color = 'black',fill_color = "black", legend_label="Confirmation Marker")
    p.legend.location = "top_left"
    p.legend.background_fill_color = "white"
    p.legend.background_fill_alpha = 0.2
    
    return p

def areas_plot(p, x_hh_map, y_hh, width_areas):
    height_bull = 0.15*y_hh[-1]
    height_bear = 0.15*y_hh[-1]
    height_crash = y_hh[-1] - height_bear

    p.rect(x = x_hh_map[-1] + int(width_areas/2.), y = y_hh[-1] + height_bull/2., #setting center bullish zone
           width = width_areas, height = height_bull,
           color='green', fill_color = "green", alpha = 0.3)
    p.rect(x = x_hh_map[-1] + int(width_areas/2.), y = y_hh[-1] - height_bull/2., #setting center bearish zone
           width = width_areas, height = height_bear,
           color='red', fill_color = "red", alpha = 0.3)
    p.rect(x = x_hh_map[-1] + int(width_areas/2.), y = y_hh[-1] - height_bear - height_crash/2,
           width = width_areas, height = height_crash,
           color='black', fill_color = "black", alpha = 0.3)
    
    return p

def plot_sample(dfi, i, order, K, lf):
    # convert datetime to timestamp
    dfi['datetime'] = dfi.index
    dfi['timestamp'] = dfi.index.to_series().apply(lambda x: x.timestamp()*1000) # to milliseconds
    dfi.set_index('timestamp', inplace = True)
    
    # getting variables of reference
    x_hh_map, x_hh_conf_map, length, y_hh, y_hh_conf, candle_seconds = prep_plot(dfi)
    
    curdoc().theme = 'light_minimal'
    TOOLS = "pan,wheel_zoom,box_zoom,reset,save"
    
    p = figure(
        title = f"Higher High Sample - {dfi.datetime.iloc[0]} to {dfi.datetime.iloc[-1]}",
        x_axis_type = "datetime",
        tools = TOOLS,
        plot_width = 700
    )
    
    p.grid.grid_line_alpha = 1 # intensity of grid appearance
    
    candle_milliseconds = candle_seconds*1000 # bokeh plots in datetimes are always in ms
    width_candles = candle_milliseconds*0.8 # candle width, if 1.0 => no distance between candles
    p = candles_plot(p, dfi, width_candles)
    p = objects_plot(p, x_hh_map, x_hh_conf_map, y_hh, y_hh_conf)
    
    width_areas = candle_milliseconds*length*lf # width of prediction areas
    p = areas_plot(p, x_hh_map, y_hh, width_areas)

    p.y_range = Range1d(min(dfi['low']), max(dfi['high']))
    
    return p

        
# ~~~~~~~~~~ PLOTTING METHODS WITH PREDICTIONS PROBABILITIES ~~~~~ #
        
def predictions_plot(p, y_pred_proba, center):
    tab = "\t\t\t"
    
    # Text and position
    text = f"My smoked wisdom is ...\n bull:  {tab}{round(y_pred_proba[i][1]*100,1)}%\n bear: {tab}{round(y_pred_proba[i][0]*100,1)}%\n crash:{tab}{round(y_pred_proba[i][2]*100,1)}%"
    
    # Create a Label glyph and add it to your plot:
    mytext = Label(
        x = center[0], y = center[1],
        text = text, text_font_size = "12pt",
        background_fill_color = "white", background_fill_alpha = 0.5,
        border_line_color = 'purple',
        border_line_width = 1,
        text_align = 'center'
    )
    
    p.add_layout(mytext)
    
    url = "https://raw.githubusercontent.com/deepotrus/smoking-lemur/main/violet-lemur.png"
    p.image_url(
        x = center[0], y = center[1],
        url = value(url), alpha=1.0, anchor = "bottom_center",
        w=150, h=150, w_units="screen", h_units="screen"
    )
    return p


def plot_sample_prediction(y_pred_proba, dfi, i, order, K, lf, name, pair):
    # convert datetime to timestamp
    dfi['datetime'] = dfi.index
    dfi['timestamp'] = dfi.index.to_series().apply(lambda x: x.timestamp()*1000) # to milliseconds
    dfi.set_index('timestamp', inplace = True)
    
    # getting variables of reference
    x_hh_map, x_hh_conf_map, length, y_hh, y_hh_conf, candle_seconds = prep_plot(dfi)
    
    curdoc().theme = 'light_minimal'
    TOOLS = "pan,wheel_zoom,box_zoom,reset,save"
    
    p = figure(
        title = f"{name.upper()}{pair.upper()} - {dfi.datetime.iloc[0]} to {dfi.datetime.iloc[-1]}",
        x_axis_type = "datetime",
        tools = TOOLS,
        plot_width = 700
    )
    
    p.grid.grid_line_alpha = 1 # intensity of grid appearance
    
    candle_milliseconds = candle_seconds*1000 # bokeh plots in datetimes are always in ms
    width_candles = candle_milliseconds*0.8 # candle width, if 1.0 => no distance between candles
    p = candles_plot(p, dfi, width_candles)
    p = objects_plot(p, x_hh_map, x_hh_conf_map, y_hh, y_hh_conf)
    
    width_areas = candle_milliseconds*length*lf # width of prediction areas
    p = areas_plot(p, x_hh_map, y_hh, width_areas)
    
    center = [dfi.index[int(len(dfi)/2)], min(dfi['low'])]
    p = predictions_plot(p, y_pred_proba, center)
    p.y_range = Range1d(min(dfi['low']), max(dfi['high']))
    
    return p

        
        

##################################################
##################################################
# ~~~~~~~~~~~~~~~~~~~~ MAIN ~~~~~~~~~~~~~~~~~~~~~#
##################################################
##################################################

st.write("# Crypto Higher Highs Classifier")
st.write("## by Andrei Potra feat. ~Smoking Lemur")
st.write("This is a basic web app for visualizing crypto related data using trading view datafeed. The app is willing to show the last higher high patterns for the desired cryptocurrency in order to check what could happen. To improve the experience, a very nice and friendly lemur is willing to share with you his wisdom by smoking something i don't really know, showing you the probabilities of the price to rise, fall, or even crash... Enjoy!")

name = st.text_input("Insert crypto (e.g. btc, eth, doge): ", "btc")
pair = st.text_input("Insert pair exchange (or dominance if crypto is btc), e.g. (usd, usdt, .D): ", "usdt")
market = st.text_input("Choose market:", "BINANCE")
timeframe = st.text_input("Choose Timeframe: (e.g. 1d, 4h, 5m)", "4h")

tv = TvDatafeed()

df, w = get_market_data(name, pair, timeframe, market)
p = plot_big_timeseries(df, w, name, pair)
st.bokeh_chart(p, use_container_width=True)



# PARAMETERS for SAMPLES
st.write(f"## Price Prediction with ~Smoking Lemur")
st.write("The first parameter for our model is a multiplier. For example, if the pattern took 4 days to form and the multiplier is x2, then the prediction will have a validity for the next 8 days.")
lf = st.number_input("Insert lookforward parameter (e.g. x1, x2, x3):", min_value=1, max_value=3, value=3)

st.write("The second parameter for our model is the order of the relative maximum. For example, if the order is 3, it means that a maximum in price is determined by looking at the past & next 3 candles.")
order = st.number_input('Insert order of relative max:', min_value=2, max_value=20, value=4)

st.write("The third parameter for our model is a 'give me some breath' parameter, which lets the model wait for some time before starting to look for new patterns. For example, if the value is 10 candles, the model will wait 10 candles.")
tau = st.number_input("Insert temporal distance between samples (e.g. 10 candles):", min_value=6, max_value=100, value=12)

K = 2 # consecutive highs

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

st.write(f"Hey, Smoking Lemur here... i analyzed the market and found {len(df_hh)} samples! Now i will try to learn at my best in order to get nice predictions for you :)")

##################################################
# ~~~~~~~~~~ FEATURE EXTRACTION ~~~~~~~~~~~~~~~~~#
##################################################

target = list()

pctv_price = list()
n_candles = list()
pctv_valley = list()

for dfi in df_hh:
    price = np.array(dfi.close)
    dates = dfi.index

    hh = getHigherHighs(price, order, K)
    hh_idx = list(hh[0]) # convert deque to list

    # setting coordinates for the segment of the pattern
    x_hh = hh_idx
    y_hh = dfi.close.iloc[hh_idx].to_list()
    
    # Get target classes
    target.append( get_target_class(price, x_hh, y_hh, order) )
    
    # Build features
    pctv_price.append( ( y_hh[-1] - y_hh[0] ) / y_hh[0] )
    n_candles.append( x_hh[-1] - x_hh[0] )
    pctv_valley.append( ( price[x_hh[0]:x_hh[-1]].min() - y_hh[0] ) / y_hh[0] )

columns = ['pctv_price','n_candles','pctv_valley','target']
Xy = pd.DataFrame(zip(pctv_price,n_candles,pctv_valley,target), columns = columns)
X, y = get_X_y(Xy)


##################################################
# ~~~~~~~~~~ RANDOM FOREST ~~~~~~~~~~~~~~~~~~~~~~#
##################################################

split_point = int(X.shape[0]*0.80)
X_train = X[0:split_point]; X_test = X[split_point:]
y_train = y[0:split_point]; y_test = y[split_point:]

model = RandomForestClassifier(n_estimators = 20, max_depth=5, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

st.write("Here! I made it, i trained on 80 samples and we'll use the latest ones to test me >:) Go ahead and choose a number from -1 to -50, -1 means the most recent pattern i found on the market.")
i = st.number_input("Insert a index for some of the latest samples (e.g. -1, -2, -3 ...): ", min_value=-50, max_value=-1 , value=-1)
p = plot_sample_prediction(y_pred_proba, df_hh[i].copy(), i, order, K, lf, name, pair)
st.bokeh_chart(p, use_container_width=True)
st.write("My friend, remember this is not financial advice, the model is actually pretty stupid, the purpose of this project is to create web app with Streamlit simulating the offering of a simple service")
