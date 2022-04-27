import matplotlib
import streamlit as st

from azureml.core import Workspace
from azureml.core import Model
from azureml.core import Dataset

from azureml.core.authentication import ServicePrincipalAuthentication

import json
import pandas as pd
import pickle
import datetime

from fmputils import FMP

import quantstats as qs

import plotly.graph_objects as go

# imports and environment setup
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor


def calcZScores(X):
    '''
    Calculate Altman Z'' scores 1995
    '''
    Z = pd.DataFrame()
    Z['Z score'] = 3.25 \
        + 6.51 * X['(CA-CL)/TA']\
        + 3.26 * X['RE/TA']\
        + 6.72 * X['EBIT/TA']\
        + 1.05 * X['Book Equity/TL']
    return Z


aml_auth = ServicePrincipalAuthentication(tenant_id=st.secrets['AML_TENANT_ID'],
                                                  service_principal_id=st.secrets['AML_PRINCIPAL_ID'],
                                                  service_principal_password=st.secrets['AML_PRINCIPAL_PASSWORD'])

ws = Workspace.get(name=st.secrets['AML_WORKSPACE_NAME'],
                       resource_group=st.secrets['AML_RESOURCE_GROUP'],
                       auth=aml_auth,
                       subscription_id=st.secrets['AML_SUBSCRIPTION_ID'])

fmp = FMP(st.secrets['FMP_API_KEY'])

st.title('AI Stock Picker')

# Select Model
# -> Model Statistics
# -> Stocks
# -> Backtest (+ Trailing Stop Params + Report)
# -> Benchmark Report (SPY vs. MSCI)
# -> sector/industry split
model_meta = {}
model_version = {}
for model in Model.list(ws):
    # if model name starts with "AIInvestor-"
    if model.name.startswith('AIInvestor-'):
        # ensure metadata of only the latest model is loaded
        if model.name in model_meta:
            if model.version > model_version[model.name]:
                model_meta[model.name] = model.properties
                model_version[model.name] = model.version
        else:
            model_meta[model.name] = model.properties
            model_version[model.name] = model.version
model_name = st.selectbox('Model', model_meta.keys())

sections = {'explain':False, 'backtest':False}

with st.expander('Model Metadata'):
    # retrieve the available models from AzureML
    # List registered models
    h1,h2 = st.columns([3,3])
    h1.subheader('Model Metrics')
    h2.subheader('Stocks Selected')
    c1,c2,c3,c4 = st.columns([1,1,1,3])
    c1.metric('Records trained', model_meta[model_name]['records_trained'])
    c2.metric('Features', model_meta[model_name]['no_features'])
    c3.metric('Shares', model_meta[model_name]['unique_stocks'])
    c1.metric('Years', model_meta[model_name]['years_trained'])
    c2.metric('CAGR', str(round(float(model_meta[model_name]['cagr'])*100,1)) + "%")
    c3.metric('Sharpe Ratio', round(float(model_meta[model_name]['sharpe']),2))

    stocks_benchmark = pd.DataFrame()
    stock_selected = json.loads(model_meta[model_name]['stock_selection'])
    for yr in stock_selected:
        stocks_yr = pd.DataFrame(stock_selected[yr])
        stocks_yr.columns = [yr]
        stocks_benchmark = stocks_benchmark.join(stocks_yr, how="outer")
    c4.table(stocks_benchmark)

    # todo: Model Explainer

    if st.button('Show Backtest'):
        sections['backtest'] = True       

if sections['backtest'] == True:
    with st.expander('Model Backtest'):
        returns_history = ""
        for year in stocks_benchmark.columns:
            top_10_stocks = stocks_benchmark[year].to_list()
            top_10_stocks = [item for item in top_10_stocks if not(pd.isnull(item)) == True]
            returns = fmp.getReturns(top_10_stocks, f"{str(int(year)+1)}-04-01", f"{str(int(year)+2)}-03-31")
            if isinstance(returns_history, str):
                returns_history = returns
            else:
                returns_history = returns_history.append(returns)

        st.subheader('Backtesting Results')
        st.pyplot(qs.plots.returns(returns_history, benchmark="SPY", show=False))
        st.pyplot(qs.plots.log_returns(returns_history, benchmark="SPY", show=False))
        st.pyplot(qs.plots.monthly_returns(returns_history, show=False))

        qs.reports.html(returns_history, "SPY", output=".", download_filename=f"{model_name}_backtest.html")
        with open(f"{model_name}_backtest.html", "rb") as file:
            btn = st.download_button(
                    label="Download Backtesting Report",
                    data=file,
                    file_name=f"{model_name}_backtest.html",
                    mime="text/html"
                )

dp1,dp2 = st.columns([1,1])

currentDateTime = datetime.datetime.now()
date = currentDateTime.date()
year = date.strftime("%Y")

start_date = dp1.date_input(
    "Report Start Date",
    datetime.date(int(year)-1, 12, 31))
end_date = dp2.date_input(
    "Report End Date", start_date)

# Prediction
# -> selected stocks
# -> stock statistics

if st.button('Pick Stocks'):

    status = st.empty()
    status.write("Loading Model...")

    Model(ws, model_name).download(target_dir='.', exist_ok=True)
    pipeline = pickle.load(open("model.pkl", "rb" ))

    status.write("Loading Data...")

    dsname = model_meta[model_name]['dataset']
    X = Dataset.get_by_name(ws, f'{dsname}-predictors_clipped').to_pandas_dataframe()
    tickers = Dataset.get_by_name(ws, f'{dsname}-tickers_dates').to_pandas_dataframe()
    
    bool_list = X['date'].between(
        pd.to_datetime(start_date, format='%Y-%m-%d', errors='coerce'),
        pd.to_datetime(end_date, format='%Y-%m-%d', errors='coerce'))    
    X = X[bool_list]
    tickers = tickers[bool_list] 

    X = X.drop(['date', 'date_added'], axis=1)

    status.write(f'Picking the most valuable stocks from {tickers.shape[0]} companies...')

    y_pred = pipeline.predict(X)
    y_pred = pd.DataFrame(y_pred)
    z = calcZScores(X)
    zbl = (z['Z score'].reset_index(drop=True) > 2) 
    Final_Predictions = pd.DataFrame()
    Final_Predictions[['Ticker','Report Date']] = \
    tickers[['Ticker1','Date1']].reset_index(drop=True)\
                            [zbl].reset_index(drop=True) # Ticker -> Ticker1, Date -> Date1
    
    Final_Predictions['Perf. Score'] = y_pred.reset_index(drop=True)\
                                    [zbl].reset_index(drop=True)
    
    Final_Predictions = Final_Predictions.sort_values(by='Perf. Score', 
                                        ascending=False)\
                                .head(20)
                                #.reset_index(drop=True).head(20)

    
    candidates = pd.DataFrame()

    for row in range(len(Final_Predictions)):
        ticker = Final_Predictions.iloc[row]['Ticker']
        profile = fmp.getFMPData(f'https://financialmodelingprep.com/api/v3/profile/{ticker}')[0]
        candidate = pd.DataFrame({'Ticker': ticker, 'ISIN':profile['isin'],
            'Name': profile['companyName'], 'Industry':profile['industry'],
            'Sector':profile['sector'], 'Country':profile['country'],
            'Price':profile['price'], 'DCF':profile['dcf'],
            'Report Date':Final_Predictions.iloc[row]['Report Date'],
            'Perf. Score':Final_Predictions.iloc[row]['Perf. Score']}, index=[0])
        candidates = pd.concat([candidates, candidate])
        candidates.dropna(subset=['ISIN'], inplace=True)
        candidates.drop_duplicates(subset=['Name'], keep='first', inplace=True, ignore_index=True)
    status.write(f'The top 10 stocks are:')
    st.table(candidates.loc[:9, ['ISIN', 'Name', 'Price', 'DCF', 'Report Date', 'Perf. Score']])

    year = str(start_date).split('-')[0]
    top_10_stocks = candidates['Ticker'].to_list()
    top_10_stocks = [item for item in top_10_stocks if not(pd.isnull(item)) == True]
    returns = fmp.getReturns(top_10_stocks, f"{str(int(year)+1)}-04-01", f"{str(int(year)+2)}-03-31")

    st.subheader(f'Portfolio Performance since {str(int(year)+1)}-04-01')
    st.pyplot(qs.plots.returns(returns, benchmark="SPY", show=False))
    st.pyplot(qs.plots.log_returns(returns, benchmark="SPY", show=False))
    st.pyplot(qs.plots.monthly_returns(returns, show=False))


    # Statistics    
    sector_count = candidates.loc[:9, ['Ticker', 'Sector']].groupby('Sector').count()
    sector_count['Sector'] = sector_count.index
    country_count = candidates.loc[:9, ['Ticker', 'Country']].groupby('Country').count()
    country_count['Country'] = country_count.index

    # labels = sector_count['Sector'].to_list()
    # values = sector_count['Ticker'].to_list()
    # fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5, title='Sector')])
    # fig.update_layout(showlegend=False)
    # st.plotly_chart(fig)

    # labels = country_count['Country'].to_list()
    # values = country_count['Ticker'].to_list()
    # fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3, title='Country')])
    # fig.update_layout(showlegend=False)
    # st.plotly_chart(fig)

    print(sector_count['Sector'].to_list())
    print(country_count['Country'].to_list())
    
    fig =go.Figure(go.Sunburst(
    labels=sector_count['Sector'].to_list(),
    parents=country_count['Country'].to_list(),
    values=[1,1,1,1,1,1,1,1,1,1],
    ))
    fig.update_layout(margin = dict(t=0, l=0, r=0, b=0))
    st.plotly_chart(fig)


    import shap
    import matplotlib.pyplot as plt

    print(model_meta[model_name])
    if model_meta[model_name]['model'] == 'KNeighborsRegressor':
        explainer = shap.KernelExplainer(pipeline.named_steps['KNeighborsRegressor'])
        observations = pipeline.named_steps['Power Transformer'].transform(X)
    else:
        explainer = shap.Explainer(pipeline)
        observations = X
    shap_values = explainer.shap_values(observations)

    shap.summary_plot(shap_values, X, show=False)
    fig = plt.gcf()
    st.pyplot(fig)

    data_for_prediction = X.iloc[0]
    data_for_prediction_array = data_for_prediction.values.reshape(1, -1)

    shap_values = explainer.shap_values(data_for_prediction)
    shap.force_plot(explainer.expected_value, shap_values, data_for_prediction, matplotlib=True, show=False)
    fig = plt.gcf()
    st.pyplot(fig)