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

@st.experimental_memo
def getModelData(_ws):
    model_meta = {}
    model_version = {}
    for model in Model.list(_ws):
        # if model name starts with "AIInvestor-"
        if model.name.startswith('AIInvestor-'):
            # ensure metadata of only the latest model is loaded
            if model.name in model_meta:
                if model.version > model_version[model.name]:
                    model_meta[model.name] = model.properties
                    model_version[model.name] = model.version
                    if model.run_id in model_meta:
                        model_meta[model.name]['run_id'] = model.run_id
            else:
                model_meta[model.name] = model.properties
                model_version[model.name] = model.version
                if model.run_id in model_meta:
                        model_meta[model.name]['run_id'] = model.run_id

    return model_meta, model_version

@st.experimental_memo
def loadModel(_ws, model_name, model):
    Model(_ws, model_name).download(target_dir='.', exist_ok=True)
    pipeline = pickle.load(open(f"{model}.pkl", "rb" ))
    return pipeline

def getPredictors(_ws, dsname):
    X = Dataset.get_by_name(_ws, f'{dsname}-predictors_clipped').to_pandas_dataframe()
    tickers = Dataset.get_by_name(_ws, f'{dsname}-tickers_dates').to_pandas_dataframe()
    return X, tickers

@st.experimental_memo
def getSHAPSummary(_ws, run_id, model):
    for file in _ws.get_run(run_id).get_file_names():
       # if file starts with Models/{model}/SHAP Summary then download it
        if file.startswith(f'Models/{model}/SHAP Summary'):
            _ws.get_run(run_id).download_file(file, f'shap_summary_{model}_{run_id}.png')
    return f'shap_summary_{model}_{run_id}.png'

@st.experimental_memo
def getModelComparison(_ws, run_id):
    for file in _ws.get_run(run_id).get_file_names():
       # if file starts with Models/{model}/SHAP Summary then download it
        if file.startswith(f'Models/Model Comparison'):
            _ws.get_run(run_id).download_file(file, f'model_comparison_{run_id}.png')
    best_model = _ws.get_run(run_id).properties['best_model']
    return f'model_comparison_{run_id}.png', best_model

aml_auth = ServicePrincipalAuthentication(tenant_id=st.secrets['AML_TENANT_ID'],
                                                  service_principal_id=st.secrets['AML_PRINCIPAL_ID'],
                                                  service_principal_password=st.secrets['AML_PRINCIPAL_PASSWORD'])

ws = Workspace.get(name=st.secrets['AML_WORKSPACE_NAME'],
                       resource_group=st.secrets['AML_RESOURCE_GROUP'],
                       auth=aml_auth,
                       subscription_id=st.secrets['AML_SUBSCRIPTION_ID'])

fmp = FMP(st.secrets['FMP_API_KEY'])

st.title('AI Stock Picker')

headercol1,headercol2 = st.columns([4,1])
if headercol2.button("Clear Cache"):
    # Clear values from *all* memoized functions:
    st.experimental_memo.clear()

model_meta, model_version = getModelData(ws)

model_name = st.selectbox('Model', model_meta.keys())

sections = {'explain':False, 'backtest':False}

# retrieve the available models from AzureML
# List registered models
h1,h2 = st.columns([3,3])
h1.subheader('Model Metrics')
h2.subheader('Model Explanation')
c1,c2,c3,c4 = st.columns([1,1,1,3])
c1.metric('Records trained', model_meta[model_name]['records_trained'])
c2.metric('Features', model_meta[model_name]['no_features'])
c3.metric('Shares', model_meta[model_name]['unique_stocks'])
c1.metric('Years', model_meta[model_name]['years_trained'])
c2.metric('CAGR', str(round(float(model_meta[model_name]['cagr'])*100,1)) + "%")
c3.metric('Sharpe Ratio', round(float(model_meta[model_name]['sharpe']),2))

print(model_meta[model_name]['dataset'])

if model_meta[model_name]['model'] != 'KNeighborsRegressor':
    c4.image(getSHAPSummary(ws, model_meta[model_name]['run_id'], model_meta[model_name]['model']))
else:
    c4.caption('KNeighborsRegressor cannot be explained')
st.subheader('Stocks Selected')
stocks_benchmark = pd.DataFrame()
stock_selected = json.loads(model_meta[model_name]['stock_selection'])
for yr in stock_selected:
    stocks_yr = pd.DataFrame(stock_selected[yr])
    stocks_yr.columns = [yr]
    stocks_benchmark = stocks_benchmark.join(stocks_yr, how="outer")
st.table(stocks_benchmark)

st.subheader('Model Comparison')
model_comparison, best_model = getModelComparison(ws, model_meta[model_name]['run_id'])
st.caption(f'The best model is {best_model}')
st.image(model_comparison)

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

status = st.empty()

status.write("Loading Data...")

dsname = model_meta[model_name]['dataset']

X, tickers = getPredictors(ws, dsname)

bool_list = X['date'].between(
    pd.to_datetime(start_date, format='%Y-%m-%d', errors='coerce'),
    pd.to_datetime(end_date, format='%Y-%m-%d', errors='coerce'))    
X = X[bool_list]
tickers = tickers[bool_list] 

#X = X.drop(['date', 'date_added'], axis=1)
X = X.drop(['date'], axis=1)

with st.expander('Dataset Statistics'):
    selected_features = st.multiselect('Features', list(X.columns))
    try:
        st.write(X[selected_features].describe())
    except:
        st.caption('No Features Selected')

status.write('')

if st.button('Pick Stocks'):

    status.write("Loading Model...")

    pipeline = loadModel(ws, model_name, model_meta[model_name]['model'])

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
    Final_Predictions[[x for x in X.columns]] = X[[x for x in X.columns]].reset_index(drop=True)[zbl].reset_index(drop=True)
    
    Final_Predictions = Final_Predictions.sort_values(by='Perf. Score', 
                                        ascending=False)\
                                .reset_index(drop=True).head(20)
    
    candidates = pd.DataFrame()

    for row in range(len(Final_Predictions)):
        ticker = Final_Predictions.iloc[row]['Ticker']
        profile = fmp.getFMPData(f'https://financialmodelingprep.com/api/v3/profile/{ticker}')[0]
        
        Final_Predictions.loc[Final_Predictions.index[row],'ISIN'] = profile['isin']
        Final_Predictions.loc[Final_Predictions.index[row],'Name'] = profile['companyName']
        Final_Predictions.loc[Final_Predictions.index[row],'Industry'] = profile['industry']
        Final_Predictions.loc[Final_Predictions.index[row],'Sector'] = profile['sector']
        Final_Predictions.loc[Final_Predictions.index[row],'Country'] = profile['country']
        Final_Predictions.loc[Final_Predictions.index[row],'Price'] = profile['price']
        Final_Predictions.loc[Final_Predictions.index[row],'Price Chg'] = profile['changes']
        Final_Predictions.loc[Final_Predictions.index[row],'Currency'] = profile['currency']
        Final_Predictions.loc[Final_Predictions.index[row],'Description'] = profile['description']        
        candidates = Final_Predictions.copy()
        
    #     candidate = pd.DataFrame({'Ticker': ticker, 'ISIN':profile['isin'],
    #         'Name': profile['companyName'], 'Industry':profile['industry'],
    #         'Sector':profile['sector'], 'Country':profile['country'],
    #         'Price':profile['price'], 'DCF':profile['dcf']}, index=[0])
    #     candidates = pd.concat([candidates, candidate])
    # candidates = candidates.merge(Final_Predictions, how='left', on='Ticker')
    # candidates.dropna(subset=['ISIN'], inplace=True)
    # candidates.drop_duplicates(subset=['Name'], keep='first', inplace=True, ignore_index=True)
    status.write('')
    
    predcol1, predcol2 = st.columns([1,1])
    predcol1.subheader(f'The top 10 stocks are:')
    predcol1.table(candidates.loc[:9, ['ISIN', 'Name']])

    year = str(start_date).split('-')[0]
    top_10_stocks = candidates['Ticker'].to_list()
    top_10_stocks = [item for item in top_10_stocks if not(pd.isnull(item)) == True]
    returns = fmp.getReturns(top_10_stocks, f"{str(int(year)+1)}-04-01", f"{str(int(year)+2)}-03-31")

    predcol2.subheader(f'Portfolio Performance since {str(int(year)+1)}-04-01')
    predcol2.pyplot(qs.plots.returns(returns, benchmark="SPY", show=False))
    #predcol2.pyplot(qs.plots.log_returns(returns, benchmark="SPY", show=False))
    predcol2.pyplot(qs.plots.monthly_returns(returns, show=False))


    import shap
    import matplotlib.pyplot as plt

    if model_meta[model_name]['model'] != 'KNeighborsRegressor':
        explainer = shap.Explainer(pipeline)
        observations = X

#### LINE BY LINE
    cmeta1,cmeta2 = {},{}
    for candidate in range(10):
        st.markdown('---')
        st.subheader(f'{candidate} - {candidates.iloc[candidate]["Name"]} ({candidates.iloc[candidate]["Ticker"]})')
        cmeta1[candidate],cmeta2[candidate] = st.columns([2,4])

        cmeta1[candidate].caption('Core Information')
        cmeta1[candidate].markdown(f'**ISIN:** {candidates.iloc[candidate]["ISIN"]}')
        cmeta1[candidate].markdown(f'**Country:** {candidates.iloc[candidate]["Country"]}')
        cmeta1[candidate].markdown(f'**Sector:** {candidates.iloc[candidate]["Sector"]}')
        cmeta1[candidate].markdown(f'**Industry:** {candidates.iloc[candidate]["Industry"]}')
        cmeta1[candidate].caption('Fundamentals:')
        cmeta1[candidate].markdown(f'**Price:** {candidates.iloc[candidate]["Price"]} {candidates.iloc[candidate]["Currency"]}')
        # add lines for EV/EBIT, ROCE, P/E
        cmeta1[candidate].markdown(f'**EV/EBIT:** {round(candidates.iloc[candidate]["EV/EBIT"],1)}')
        cmeta1[candidate].markdown(f'**P/E:** {round(candidates.iloc[candidate]["P/E"],1)}')
        cmeta1[candidate].markdown(f'**ROCE:** {round(candidates.iloc[candidate]["ROCE"]*100,1)}%')
        cmeta1[candidate].markdown(f'**Gross Profit Margin:** {round(candidates.iloc[candidate]["Gross Profit Margin"]*100,1)}%')
        cmeta1[candidate].markdown(f'**RoE:** {round(candidates.iloc[candidate]["RoE"]*100,1)}%')
        # add lines for Gross Profit Margin, P/E Valuation, EV/EBIT Valuation, Debt Ratio
        cmeta1[candidate].markdown(f'**P/E Valuation:** {round(candidates.iloc[candidate]["P/E Valuation"]*100,1)}%')
        cmeta1[candidate].markdown(f'**EV/EBIT Valuation:** {round(candidates.iloc[candidate]["EV/EBIT Valuation"]*100,1)}%')
        cmeta1[candidate].markdown(f'**Debt Ratio:** {round(candidates.iloc[candidate]["Debt Ratio"]*100,1)}%')
        
        cmeta2[candidate].caption('Description:')
        # shorten candidates.iloc[candidate]["Description"] maximum 700 characters and add '...' if it was shortened
        if len(candidates.iloc[candidate]["Description"]) > 700:
            cmeta2[candidate].markdown(f'{candidates.iloc[candidate]["Description"][:700]}...')
        else:
            cmeta2[candidate].markdown(f'{candidates.iloc[candidate]["Description"]}')
        cmeta2[candidate].caption('Further Information:')
        cmeta2[candidate].markdown(f'[Summary](https://finance.yahoo.com/quote/{candidates.iloc[candidate]["Ticker"]}/)')
        cmeta2[candidate].markdown(f'[Analyst Ratings](https://finance.yahoo.com/quote/{candidates.iloc[candidate]["Ticker"]}/analysis)')
        cmeta2[candidate].caption('Model Insights:')
        
        
        if model_meta[model_name]['model'] != 'KNeighborsRegressor':
            data_for_prediction = candidates.iloc[candidate,3:-9]
            shap_values = explainer.shap_values(data_for_prediction)
            plt.clf()
            shap.force_plot(explainer.expected_value, shap_values, data_for_prediction, text_rotation=45, matplotlib=True, show=False)
            fig = plt.gcf()
            cmeta2[candidate].pyplot(fig)


#### SUMMARY PLOT

    st.markdown('---')
    st.caption('Summary Plot for the top 10 stocks')
    if model_meta[model_name]['model'] != 'KNeighborsRegressor':
        plt.clf()
        data_for_prediction = candidates.iloc[:10,3:-9]
        shap_values = explainer.shap_values(data_for_prediction)
        shap.summary_plot(shap_values, data_for_prediction, show=False)
        fig = plt.gcf()
        st.pyplot(fig)

    st.markdown('---')
    st.caption('Summary Plot on available data')
    if model_meta[model_name]['model'] != 'KNeighborsRegressor':
        plt.clf()
        shap_values = explainer.shap_values(observations)
        shap.summary_plot(shap_values, X, show=False)
        fig = plt.gcf()
        st.pyplot(fig)

    # # Statistics    
    # sector_count = candidates.loc[:9, ['Ticker', 'Sector']].groupby('Sector').count()
    # sector_count['Sector'] = sector_count.index
    # country_count = candidates.loc[:9, ['Ticker', 'Country']].groupby('Country').count()
    # country_count['Country'] = country_count.index

    # # labels = sector_count['Sector'].to_list()
    # # values = sector_count['Ticker'].to_list()
    # # fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5, title='Sector')])
    # # fig.update_layout(showlegend=False)
    # # st.plotly_chart(fig)

    # # labels = country_count['Country'].to_list()
    # # values = country_count['Ticker'].to_list()
    # # fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3, title='Country')])
    # # fig.update_layout(showlegend=False)
    # # st.plotly_chart(fig)

    # print(sector_count['Sector'].to_list())
    # print(country_count['Country'].to_list())

    # fig =go.Figure(go.Sunburst(
    # labels=sector_count['Sector'].to_list(),
    # parents=country_count['Country'].to_list(),
    # values=[1,1,1,1,1,1,1,1,1,1],
    # ))
    # fig.update_layout(margin = dict(t=0, l=0, r=0, b=0))
    # st.plotly_chart(fig)


    