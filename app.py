import streamlit as st
import yfinance as yf
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Set page config
st.set_page_config(page_title="Stock Prediction", layout="wide")

# Define holidays in YYYY-MM-DD format
HOLIDAYS = [
    "2024-01-22", "2024-01-26", "2024-03-08", "2024-03-25", "2024-03-29", 
    "2024-04-11", "2024-04-17", "2024-05-01", "2024-05-20", "2024-06-17", 
    "2024-07-17", "2024-08-15", "2024-10-02", "2024-11-01", "2024-11-15", 
    "2024-11-20", "2024-12-25"
]

# Define stock options
STOCK_OPTIONS = {
    'Nifty': {
        'ticker': '^NSEI',
        'model_path': 'models/stock_prediction_model_nifty.joblib'
    },
    'HDFC': {
        'ticker': 'HDFCLIFE.NS',
        'model_path': 'models/stock_prediction_model_hdfc.joblib'
    },
    'Adani Green Energy': {
        'ticker': 'ADANIGREEN.NS',
        'model_path': 'models/stock_prediction_model_adanienergy.joblib'
    },
    'Adani Energy': {
        'ticker': 'ADANIENSOL.NS',
        'model_path': 'models/stock_prediction_model_adanigreen.joblib'
    },
    'Airtel': {
        'ticker': 'BHARTIARTL.NS',
        'model_path': 'models/stock_prediction_model_airtel.joblib'
    },
    'BEL': {
        'ticker': 'BEL.NS',
        'model_path': 'models/stock_prediction_model_bel.joblib'
    },
}

# Load the selected model
@st.cache_resource
def load_model(model_path):
    try:
        if os.path.exists(model_path):
            return joblib.load(model_path)
        else:
            st.error(f"Model file not found: {model_path}")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def fetch_stock_data(ticker, period='1d', interval='5m'):
    try:
        stock_data = yf.download(ticker, period=period, interval=interval)
        if stock_data.empty:
            st.error("No data received from Yahoo Finance")
            return None
        return stock_data
    except Exception as e:
        st.error(f"Error fetching stock data: {str(e)}")
        return None

def predict_next_day(data, model):
    try:
        if model is None:
            st.error("Model not loaded")
            return None
            
        # Scale the data using MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        # Reshape data for LSTM
        data_reshaped = scaled_data.reshape(1, 75, 4)
        
        # Make predictions
        predicted_values = model.predict(data_reshaped)
        
        # Inverse transform predictions
        predicted_values_rescaled = scaler.inverse_transform(predicted_values.reshape(-1, 4))
        predicted_values_rescaled = predicted_values_rescaled.reshape(75, 4)
        
        return predicted_values_rescaled
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None

def create_candlestick_chart(data, title):
    # Calculate price changes and percentages
    data['Price_Change'] = data['Close'] - data['Open']
    data['Change_Percent'] = (data['Price_Change'] / data['Open']) * 100
    data['Range'] = data['High'] - data['Low']
    data['Range_Percent'] = (data['Range'] / data['Open']) * 100
    data['Average'] = (data['High'] + data['Low']) / 2

    # Create hover text
    hovertext = []
    for idx, row in data.iterrows():
        day_name = idx.strftime('%A')
        date_str = idx.strftime('%Y-%m-%d')
        hover_str = (
            f"<b>{date_str}</b><br>" +
            f"<b>{day_name}</b><br>" +
            "──────────────<br>" +
            f"<b>Price Details</b><br>" +
            f"Open: ₹{row['Open']:,.2f}<br>" +
            f"High: ₹{row['High']:,.2f}<br>" +
            f"Low: ₹{row['Low']:,.2f}<br>" +
            f"Close: ₹{row['Close']:,.2f}<br>" +
            f"Average: ₹{row['Average']:,.2f}<br>" +
            "──────────────<br>" +
            f"<b>Changes</b><br>" +
            f"Price Change: ₹{row['Price_Change']:+,.2f}<br>" +
            f"Change: {row['Change_Percent']:+.2f}%<br>" +
            f"Range: ₹{row['Range']:,.2f} ({row['Range_Percent']:.2f}%)"
        )
        hovertext.append(hover_str)

    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        increasing_line_color='#00C805',  # Brighter green
        decreasing_line_color='#FF3737',  # Brighter red
        increasing_fillcolor='rgba(0, 200, 5, 0.4)',  # Semi-transparent green
        decreasing_fillcolor='rgba(255, 55, 55, 0.4)',  # Semi-transparent red
        text=hovertext,
        hoverinfo='text'
    )])
    
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            x=0.5,
            xanchor='center',
            font=dict(size=24, color='#2E4053', family="Arial Black")
        ),
        yaxis_title=dict(
            text='Price (₹)',
            font=dict(size=16, color='#2E4053', family="Arial")
        ),
        xaxis_title=dict(
            text='Time',
            font=dict(size=16, color='#2E4053', family="Arial")
        ),
        height=600,  # Increased height
        template='plotly_white',
        showlegend=False,
        plot_bgcolor='rgba(240,242,245,0.8)',  # Light grey background
        paper_bgcolor='white',
        margin=dict(l=60, r=60, t=100, b=60),
        hoverlabel=dict(
            bgcolor='white',
            font_size=14,
            font_family="Arial",
            bordercolor='#2E4053'
        )
    )

    # Add a range slider with styling
    fig.update_xaxes(
        rangeslider=dict(
            visible=True,
            bgcolor='#F8F9F9'
        ),
        gridcolor='rgba(189,195,199,0.5)',
        gridwidth=1,
        showgrid=True
    )
    
    # Update y-axis grid
    fig.update_yaxes(
        gridcolor='rgba(189,195,199,0.5)',
        gridwidth=1,
        showgrid=True,
        zeroline=True,
        zerolinecolor='#2E4053',
        zerolinewidth=1.5
    )
    
    return fig

def create_open_close_chart(data, title):
    # Calculate price changes and percentages
    data['Price_Change'] = data['Close'] - data['Open']
    data['Change_Percent'] = (data['Price_Change'] / data['Open']) * 100
    data['Average'] = (data['High'] + data['Low']) / 2
    data['Range'] = abs(data['Close'] - data['Open'])
    data['Range_Percent'] = (data['Range'] / data['Open']) * 100
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Create hover text for Open line
    open_hover = []
    for idx, row in data.iterrows():
        day_name = idx.strftime('%A')
        date_str = idx.strftime('%Y-%m-%d')
        hover_str = (
            f"<b>{date_str}</b><br>" +
            f"<b>{day_name}</b><br>" +
            "──────────────<br>" +
            f"<b>Open Price Details</b><br>" +
            f"Open: ₹{row['Open']:,.2f}<br>" +
            f"Average: ₹{row['Average']:,.2f}<br>" +
            "──────────────<br>" +
            f"<b>Changes</b><br>" +
            f"Change: ₹{row['Price_Change']:+,.2f}<br>" +
            f"Change %: {row['Change_Percent']:+.2f}%<br>" +
            f"Range: ₹{row['Range']:,.2f} ({row['Range_Percent']:.2f}%)"
        )
        open_hover.append(hover_str)

    # Create hover text for Close line
    close_hover = []
    for idx, row in data.iterrows():
        day_name = idx.strftime('%A')
        date_str = idx.strftime('%Y-%m-%d')
        hover_str = (
            f"<b>{date_str}</b><br>" +
            f"<b>{day_name}</b><br>" +
            "──────────────<br>" +
            f"<b>Close Price Details</b><br>" +
            f"Close: ₹{row['Close']:,.2f}<br>" +
            f"Average: ₹{row['Average']:,.2f}<br>" +
            "──────────────<br>" +
            f"<b>Changes</b><br>" +
            f"Change: ₹{row['Price_Change']:+,.2f}<br>" +
            f"Change %: {row['Change_Percent']:+.2f}%<br>" +
            f"Range: ₹{row['Range']:,.2f} ({row['Range_Percent']:.2f}%)"
        )
        close_hover.append(hover_str)
    
    # Add Open line
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Open'],
            name="Open",
            line=dict(color="#00C805", width=2.5),
            fill=None,
            text=open_hover,
            hoverinfo='text'
        ),
        secondary_y=False,
    )
    
    # Add Close line
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Close'],
            name="Close",
            line=dict(color="#FF3737", width=2.5),
            fill='tonexty',
            fillcolor='rgba(128,128,128,0.1)',
            text=close_hover,
            hoverinfo='text'
        ),
        secondary_y=False,
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            x=0.5,
            xanchor='center',
            font=dict(size=24, color='#2E4053', family="Arial Black")
        ),
        height=600,
        template='plotly_white',
        hovermode='x unified',
        plot_bgcolor='rgba(240,242,245,0.8)',
        paper_bgcolor='white',
        margin=dict(l=60, r=60, t=100, b=60),
        hoverlabel=dict(
            bgcolor='white',
            font_size=14,
            font_family="Arial",
            bordercolor='#2E4053'
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='rgba(46, 64, 83, 0.3)',
            borderwidth=1,
            font=dict(size=14, color='#2E4053')
        )
    )
    
    # Update axes
    fig.update_yaxes(
        title=dict(
            text="Price (₹)",
            font=dict(size=16, color='#2E4053', family="Arial")
        ),
        gridcolor='rgba(189,195,199,0.5)',
        gridwidth=1,
        showgrid=True,
        zeroline=True,
        zerolinecolor='#2E4053',
        zerolinewidth=1.5,
        secondary_y=False
    )
    
    fig.update_xaxes(
        title=dict(
            text="Time",
            font=dict(size=16, color='#2E4053', family="Arial")
        ),
        gridcolor='rgba(189,195,199,0.5)',
        gridwidth=1,
        showgrid=True
    )
    
    return fig

def create_open_close_chart_with_dates(data, title):
    # Calculate price changes and percentages
    data['Price_Change'] = data['Close'] - data['Open']
    data['Change_Percent'] = (data['Price_Change'] / data['Open']) * 100
    
    fig = go.Figure()

    # Add sequential numbers for x-axis
    x_seq = list(range(1, len(data) + 1))
    
    # Add traces for Open and Close values
    fig.add_trace(
        go.Scatter(
            x=x_seq,
            y=data['Open'],
            name="Open",
            line=dict(color="#26A69A", width=2),  # Matching green
            hovertemplate=
            "<b>Point %{x}</b><br>" +
            "<b>%{customdata[0]}</b><br>" +
            "<b>%{customdata[1]}</b><br><br>" +
            "Open: ₹%{y:,.2f}<br>" +
            "Change: ₹%{customdata[2]:+,.2f}<br>" +
            "Change %: %{customdata[3]:.2f}%<br>" +
            "<extra></extra>",
            customdata=np.column_stack((
                data.index.strftime('%A'),
                data.index.strftime('%Y-%m-%d'),
                data['Price_Change'],
                data['Change_Percent']
            ))
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=x_seq,
            y=data['Close'],
            name="Close",
            line=dict(color="#EF5350", width=2),  # Matching red
            hovertemplate=
            "<b>Point %{x}</b><br>" +
            "<b>%{customdata[0]}</b><br>" +
            "<b>%{customdata[1]}</b><br><br>" +
            "Close: ₹%{y:,.2f}<br>" +
            "Change: ₹%{customdata[2]:+,.2f}<br>" +
            "Change %: %{customdata[3]:.2f}%<br>" +
            "<extra></extra>",
            customdata=np.column_stack((
                data.index.strftime('%A'),
                data.index.strftime('%Y-%m-%d'),
                data['Price_Change'],
                data['Change_Percent']
            ))
        )
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        ),
        yaxis_title=dict(text='Price (₹)', font=dict(size=14)),
        xaxis_title=dict(text='Prediction Point', font=dict(size=14)),
        height=400,
        template='plotly_white',
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=80, b=50),
        hoverlabel=dict(
            bgcolor='white',
            font_size=14,
            font_family="Roboto"
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.1)',
            borderwidth=1
        ),
        hovermode='x unified',
        # Add a secondary x-axis for actual dates
        xaxis2=dict(
            overlaying="x",
            side="top",
            title=dict(text="Date & Time", font=dict(size=14)),
            showgrid=False,
            ticktext=data.index.strftime('%Y-%m-%d %H:%M').tolist(),
            tickvals=x_seq,
            tickmode="array"
        )
    )
    
    # Update axes grids
    fig.update_xaxes(gridcolor='lightgrey')
    fig.update_yaxes(gridcolor='lightgrey')
    
    return fig

def create_mini_chart(data, title):
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Close'],
            name="Price",
            line=dict(color="#2E4053", width=2),
            fill='tonexty',
            fillcolor='rgba(46,64,83,0.1)'
        )
    )
    
    # Calculate price change and percentage
    price_change = data['Close'].iloc[-1] - data['Close'].iloc[0]
    change_percent = (price_change / data['Close'].iloc[0]) * 100
    color = "#00C805" if price_change >= 0 else "#FF3737"
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><span style='color:{color}'>₹{price_change:+,.2f} ({change_percent:+.2f}%)</span>",
            x=0.5,
            xanchor='center',
            font=dict(size=14, color='#2E4053', family="Arial")
        ),
        height=200,
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='white',
        plot_bgcolor='rgba(240,242,245,0.8)',
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False
        )
    )
    
    return fig

def create_ohlc_chart(data, title):
    fig = go.Figure(data=[go.Ohlc(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        increasing_line_color='#00C805',
        decreasing_line_color='#FF3737'
    )])
    
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            x=0.5,
            xanchor='center',
            font=dict(size=24, color='#2E4053', family="Arial Black")
        ),
        yaxis_title='Price (₹)',
        height=600,
        template='plotly_white',
        plot_bgcolor='rgba(240,242,245,0.8)',
        paper_bgcolor='white'
    )
    return fig

def create_area_chart(data, title):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        fill='tozeroy',
        fillcolor='rgba(46,64,83,0.2)',
        line=dict(color='#2E4053', width=2),
        name='Close Price'
    ))
    
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            x=0.5,
            xanchor='center',
            font=dict(size=24, color='#2E4053', family="Arial Black")
        ),
        yaxis_title='Price (₹)',
        height=600,
        template='plotly_white',
        plot_bgcolor='rgba(240,242,245,0.8)',
        paper_bgcolor='white'
    )
    return fig

def create_hollow_candles(data, title):
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        increasing_line_color='#00C805',
        decreasing_line_color='#FF3737',
        increasing_fillcolor='rgba(0,0,0,0)',
        decreasing_fillcolor='rgba(255,55,55,0.8)'
    )])
    
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            x=0.5,
            xanchor='center',
            font=dict(size=24, color='#2E4053', family="Arial Black")
        ),
        yaxis_title='Price (₹)',
        height=600,
        template='plotly_white',
        plot_bgcolor='rgba(240,242,245,0.8)',
        paper_bgcolor='white'
    )
    return fig

def create_heikin_ashi(data, title):
    # Calculate Heikin-Ashi values
    ha_close = (data['Open'] + data['High'] + data['Low'] + data['Close']) / 4
    ha_open = (data['Open'].shift(1) + data['Close'].shift(1)) / 2
    ha_high = data[['High', 'Open', 'Close']].max(axis=1)
    ha_low = data[['Low', 'Open', 'Close']].min(axis=1)
    
    # Create Heikin-Ashi DataFrame
    ha_data = pd.DataFrame({
        'Open': ha_open,
        'High': ha_high,
        'Low': ha_low,
        'Close': ha_close
    }, index=data.index)
    
    fig = go.Figure(data=[go.Candlestick(
        x=ha_data.index,
        open=ha_data['Open'],
        high=ha_data['High'],
        low=ha_data['Low'],
        close=ha_data['Close'],
        increasing_line_color='#00C805',
        decreasing_line_color='#FF3737',
        increasing_fillcolor='rgba(0,200,5,0.8)',
        decreasing_fillcolor='rgba(255,55,55,0.8)'
    )])
    
    fig.update_layout(
        title=dict(
            text=f"<b>{title} (Heikin-Ashi)</b>",
            x=0.5,
            xanchor='center',
            font=dict(size=24, color='#2E4053', family="Arial Black")
        ),
        yaxis_title='Price (₹)',
        height=600,
        template='plotly_white',
        plot_bgcolor='rgba(240,242,245,0.8)',
        paper_bgcolor='white'
    )
    return fig

def get_next_trading_day(current_date):
    next_date = current_date + timedelta(days=1)
    # Skip weekends and holidays
    while next_date.weekday() >= 5 or next_date.strftime("%Y-%m-%d") in HOLIDAYS:
        next_date += timedelta(days=1)
    return next_date

# Main app
st.title('Stock Price Prediction')

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    # Stock Selection
    st.subheader("Select Stock")
    selected_stock = st.selectbox("", list(STOCK_OPTIONS.keys()))
    
    # Chart Type Selection
    st.subheader("Chart Type")
    chart_type = st.selectbox("", [
        "Candlestick", 
        "Line Chart",
        "OHLC Chart",
        "Area Chart",
        "Hollow Candles",
        "Heikin-Ashi"
    ])

    # Get stock info
    stock_info = STOCK_OPTIONS[selected_stock]
    ticker = stock_info['ticker']
    model = load_model(stock_info['model_path'])

if model is not None:
    # Fetch stock data
    stock_data = fetch_stock_data(ticker)
    
    if stock_data is not None:
        # Display the latest stock data
        latest_data = stock_data[['Open', 'High', 'Low', 'Close']].tail(75)
        
        # Predict the next day's values
        predicted_values = predict_next_day(latest_data, model)
        
        if predicted_values is not None:
            # Create prediction DataFrame with market hours
            current_date = datetime.now()
            next_trading_day = get_next_trading_day(current_date)
            
            # Display prediction date information
            st.sidebar.markdown("---")
            st.sidebar.write("### Prediction Information")
            st.sidebar.write(f"Predicting for: **{next_trading_day.strftime('%A, %d-%b-%Y')}**")
            
            # Create market hours timestamps
            market_hours = []
            current_time = datetime.combine(next_trading_day.date(), datetime.strptime("9:15", "%H:%M").time())
            end_time = datetime.combine(next_trading_day.date(), datetime.strptime("15:25", "%H:%M").time())
            
            while current_time <= end_time and len(market_hours) < 75:
                market_hours.append(current_time)
                current_time += timedelta(minutes=5)
            
            # Ensure we have enough timestamps
            while len(market_hours) < 75:
                market_hours.append(market_hours[-1] + timedelta(minutes=5))
            
            prediction_df = pd.DataFrame(predicted_values, columns=['Open', 'High', 'Low', 'Close'])
            prediction_df['Timestamp'] = market_hours[:75]  # Use only first 75 timestamps
            prediction_df.set_index('Timestamp', inplace=True)

            # Current Market Data Section
            current_day = latest_data.index[-1].strftime('%A, %d-%b-%Y')
            st.subheader(f'Current Market Data - {selected_stock} for {current_day}')
            current_col1, current_col2 = st.columns([3, 1])

            with current_col2:
                st.metric("Latest Open", f"₹{latest_data['Open'].iloc[-1]:.2f}")
                st.metric("Latest High", f"₹{latest_data['High'].iloc[-1]:.2f}")
                st.metric("Latest Low", f"₹{latest_data['Low'].iloc[-1]:.2f}")
                st.metric("Latest Close", f"₹{latest_data['Close'].iloc[-1]:.2f}")

            with current_col1:
                if chart_type == 'Candlestick':
                    st.plotly_chart(create_candlestick_chart(latest_data, f'Current Market Data - {selected_stock}'), use_container_width=True)
                elif chart_type == 'OHLC Chart':
                    st.plotly_chart(create_ohlc_chart(latest_data, f'Current Market Data - {selected_stock}'), use_container_width=True)
                elif chart_type == 'Area Chart':
                    st.plotly_chart(create_area_chart(latest_data, f'Current Market Data - {selected_stock}'), use_container_width=True)
                elif chart_type == 'Hollow Candles':
                    st.plotly_chart(create_hollow_candles(latest_data, f'Current Market Data - {selected_stock}'), use_container_width=True)
                elif chart_type == 'Heikin-Ashi':
                    st.plotly_chart(create_heikin_ashi(latest_data, f'Current Market Data - {selected_stock}'), use_container_width=True)
                else:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=latest_data.index, y=latest_data['Close'], mode='lines', name='Close Price'))
                    fig.update_layout(title=f'Current Market Data - {selected_stock}',
                                   yaxis_title='Price (₹)',
                                   xaxis_title='Time',
                                   height=500)
                    st.plotly_chart(fig, use_container_width=True)

            # Add some spacing
            st.markdown("---")

            # Predicted Market Data Section
            next_day = prediction_df.index[0].strftime('%A, %d-%b-%Y')
            st.subheader(f'Predicted Market Data - {selected_stock} for {next_day}')
            pred_col1, pred_col2 = st.columns([3, 1])

            with pred_col2:
                st.metric("Predicted Open", f"₹{prediction_df['Open'].iloc[0]:.2f}")
                st.metric("Predicted High", f"₹{prediction_df['High'].iloc[0]:.2f}")
                st.metric("Predicted Low", f"₹{prediction_df['Low'].iloc[0]:.2f}")
                st.metric("Predicted Close", f"₹{prediction_df['Close'].iloc[0]:.2f}")

            with pred_col1:
                if chart_type == 'Candlestick':
                    st.plotly_chart(create_candlestick_chart(prediction_df, f'Predicted Market Data - {selected_stock}'), use_container_width=True)
                elif chart_type == 'OHLC Chart':
                    st.plotly_chart(create_ohlc_chart(prediction_df, f'Predicted Market Data - {selected_stock}'), use_container_width=True)
                elif chart_type == 'Area Chart':
                    st.plotly_chart(create_area_chart(prediction_df, f'Predicted Market Data - {selected_stock}'), use_container_width=True)
                elif chart_type == 'Hollow Candles':
                    st.plotly_chart(create_hollow_candles(prediction_df, f'Predicted Market Data - {selected_stock}'), use_container_width=True)
                elif chart_type == 'Heikin-Ashi':
                    st.plotly_chart(create_heikin_ashi(prediction_df, f'Predicted Market Data - {selected_stock}'), use_container_width=True)
                else:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=prediction_df.index, y=prediction_df['Close'], mode='lines', name='Predicted Close'))
                    fig.update_layout(title=f'Predicted Market Data - {selected_stock}',
                                   yaxis_title='Price (₹)',
                                   xaxis_title='Time',
                                   height=500)
                    st.plotly_chart(fig, use_container_width=True)

            # Add Open-Close comparison charts
            st.subheader(f'Predicted Open-Close Comparisons - {selected_stock}')
            
            # Market Hours View
            st.write("### Market Hours View (9:15 AM - 3:25 PM)")
            st.plotly_chart(create_open_close_chart(prediction_df, f'Predicted Values (Market Hours) - {selected_stock}'), use_container_width=True)
            
            # Actual Timeline View
            st.write("### Actual Timeline View")
            st.plotly_chart(create_open_close_chart_with_dates(prediction_df, f'Predicted Values (Actual Timeline) - {selected_stock}'), use_container_width=True)

            # Display detailed data tables
            st.markdown("---")
            st.subheader('Detailed Data')
            st.write("### Actual Data")
            st.dataframe(latest_data)
            st.write(f"### Predicted Data for {next_trading_day}")
            st.dataframe(prediction_df)
