import streamlit as st 
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go


START = "2014-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction System")
stocks = ("AAPL","GOOG","MSFT","GME")
selected_stock = st.selectbox("Select Dataset for Prediction", stocks)

n_years = st.slider("Years of Prediction:", 1 , 4)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load data...")
data = load_data(selected_stock)
data_load_state.text("Loading data... Done")


st.subheader('Raw data')
st.write(data.tail())


fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'],name='stock_open'))
fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'],name='stock_close',line = dict(color = 'green')))
fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
st.plotly_chart(fig)


# Forecasting
df_train = data[['Date','Close']]

# prophet accept data in this format
df_train = df_train.rename(columns={"Date":"ds","Close":"y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forcast data')
st.write(forecast.tail())


st.write('Forecast data')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('Forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)