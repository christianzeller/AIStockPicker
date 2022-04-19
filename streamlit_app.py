import streamlit as st

from azureml.core import Workspace
from azureml.core import Model

from azureml.core.authentication import ServicePrincipalAuthentication

import json
import pandas as pd

from fmputils import FMP

import quantstats as qs


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

with st.expander('Select a trained model'):
    # retrieve the available models from AzureML
    # List registered models
    model_meta = {}
    for model in Model.list(ws):
        # if model name starts with "AIInvestor-"
        if model.name.startswith('AIInvestor-'):
            model_meta[model.name] = model.properties
    model_name = st.selectbox('Model', model_meta.keys())
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


# Select Dataset
# -> Dataset Statistics
# -> Select Date Range

with st.expander('Dataset Selection'):
    st.write('tbd')

# Prediction
# -> selected stocks
# -> stock statistics