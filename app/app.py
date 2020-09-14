import pandas as pd
import numpy as np
import flask
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output,State
import plotly.graph_objects as go
import os
import subprocess

from datetime import datetime

import requests
import json
from sklearn import linear_model
reg = linear_model.LinearRegression(fit_intercept=True)
import pandas as pd

from scipy import signal
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output,State
from scipy import optimize
from scipy import integrate
import plotly.graph_objects as go
from get_data import get_johns_hopkins
from  build_features  import *
from  process_data import *
import plotly.express as px
import dash_bootstrap_components as dbc

print(os.getcwd())

get_johns_hopkins()
store_relational_JH_data()


test_data_reg=np.array([2,4,6])
result=get_doubling_time_via_regression(test_data_reg)
print('the test slope is: '+str(result))

pd_JH_data=pd.read_csv('../data/processed/COVID_relational_confirmed.csv',sep=';',parse_dates=[0])
pd_JH_data=pd_JH_data.sort_values('date',ascending=True).copy()


pd_result_larg=calc_filtered_data(pd_JH_data)
pd_result_larg=calc_doubling_rate(pd_result_larg)
pd_result_larg=calc_doubling_rate(pd_result_larg,'confirmed_filtered')


mask=pd_result_larg['confirmed']>100
pd_result_larg['confirmed_filtered_DR']=pd_result_larg['confirmed_filtered_DR'].where(mask, other=np.NaN)
pd_result_larg.to_csv('../data/processed/COVID_final_set.csv',sep=';',index=False)
print(pd_result_larg[pd_result_larg['country']=='Germany'].tail())



###SIR

proces_SIR_data()

df_analyse=pd.read_csv('../data/processed/COVID_small_flat_table.csv',sep=';')  
df_analyse.sort_values('date',ascending=True).head()

N0=1000000 #max susceptible population
beta=0.4   # infection spread dynamics
gamma=0.1  # recovery rate


# condition I0+S0+R0=N0
I0=df_analyse['Germany'][50]
S0=N0-I0
R0=0


ydata = np.array(df_analyse['Germany'][50:])
t=np.arange(len(ydata))

I0=ydata[0]
S0=N0-I0
R0=0

def SIR_model_t(SIR,t,beta,gamma):
    ''' Simple SIR model
        S: susceptible population
        t: time step, mandatory for integral.odeint
        I: infected people
        R: recovered people
        beta: 
        
        overall condition is that the sum of changes (differnces) sum up to 0
        dS+dI+dR=0
        S+I+R= N (constant size of population)
    
    '''
    
    S,I,R=SIR
    dS_dt=-beta*S*I/N0          
    dI_dt=beta*S*I/N0-gamma*I
    dR_dt=gamma*I
    return dS_dt,dI_dt,dR_dt



def fit_odeint(x, beta, gamma):
    '''
    helper function for the integration
    '''
    return integrate.odeint(SIR_model_t, (S0, I0, R0), t, args=(beta, gamma))[:,1] # we only would like to get dI

popt=[0.4,0.1]
fit_odeint(t, *popt)

popt, pcov = optimize.curve_fit(fit_odeint, t, ydata)
perr = np.sqrt(np.diag(pcov))
    
print('standard deviation errors : ',str(perr), ' start infect:',ydata[0])
print("Optimal parameters: beta =", popt[0], " and gamma = ", popt[1])

fitted=fit_odeint(t, *popt)




#Visualize
df_input_large=pd.read_csv('../data/processed/COVID_final_set.csv',sep=';')
fig = go.Figure()

app = dash.Dash(external_stylesheets=[dbc.themes.COSMO])
server=app.server
app.layout = html.Div([

    dcc.Markdown('''
    # **ENTERPRISE DATA SCIENCE - COVID-19 DATA**
    '''),
    dcc.Markdown(''' ---
Author: Tanmay Delhikar
--- '''),
    dcc.Markdown('''
    ---
    '''),
    dcc.Markdown('''
    ## **Part 1 : All Countries Visualization**
    '''),
    dcc.Markdown('''
    ### Select one or more countries
    '''),


    dcc.Dropdown(
        id='country_drop_down',
        options=[ {'label': each,'value':each} for each in df_input_large['country'].unique()],
        value=['US', 'Germany','Italy'], # which are pre-selected
        multi=True
    ),

    dcc.Markdown('''
        ### Select Timeline of confirmed COVID-19 cases or the approximated doubling time
        '''),

    dcc.Dropdown(
    id='doubling_time',
    options=[
        {'label': 'Timeline Confirmed ', 'value': 'confirmed'},
        {'label': 'Timeline Confirmed Filtered', 'value': 'confirmed_filtered'},
        {'label': 'Timeline Doubling Rate', 'value': 'confirmed_DR'},
        {'label': 'Timeline Doubling Rate Filtered', 'value': 'confirmed_filtered_DR'},
    ],
    value='confirmed',
    multi=False
    ),

    dcc.Graph(figure=fig, id='main_window_slope'),
        dcc.Markdown('''
    ---
    '''),
    dcc.Markdown('''
    ## **Part 2 : SIR Modelling**
    '''),
    dcc.Markdown('''
    ### Select any country
    '''),
    dcc.Dropdown(
                id='country_SIR',
                options=[{'label': 'Germany', 'value': 'Germany'},
                       {'label': 'US', 'value': 'US'},
                       {'label': 'Italy', 'value': 'Italy'},
                       {'label': 'Spain', 'value': 'Spain'},
                       {'label': 'Korea, South', 'value': 'Korea, South'}],
                value='Germany',
                multi=False
            ),



    
    dcc.Graph(id='sir')
])



@app.callback(
    Output('main_window_slope', 'figure') ,
    [Input('country_drop_down', 'value'),
    Input('doubling_time', 'value')])
def update_figure(country_list,show_doubling):

    if 'DR' in show_doubling:
        my_yaxis={'type':"log",
               'title':'Approximated doubling rate over 3 days (larger numbers are better #stayathome)'
              }
    else:
        my_yaxis={'type':"log",
                  'title':'Confirmed infected people (source johns hopkins csse, log-scale)'
              }


    traces = []
    for each in country_list:

        df_plot=df_input_large[df_input_large['country']==each]

        if show_doubling=='doubling_rate_filtered':
            df_plot=df_plot[['state','country','confirmed','confirmed_filtered','confirmed_DR','confirmed_filtered_DR','date']].groupby(['country','date']).agg(np.mean).reset_index()
        else:
            df_plot=df_plot[['state','country','confirmed','confirmed_filtered','confirmed_DR','confirmed_filtered_DR','date']].groupby(['country','date']).agg(np.sum).reset_index()
       


        traces.append(dict(x=df_plot.date,
                                y=df_plot[show_doubling],
                                mode='markers+lines',
                                opacity=0.9,
                                name=each
                        )
                )

    return {
            'data': traces,
            'layout': dict (
                width=1280,
                height=720,

                xaxis={'title':'Timeline',
                        'tickangle':-45,
                        'nticks':20,
                        'tickfont':dict(size=14,color="#7f7f7f"),
                      },

                yaxis=my_yaxis
        )
    }


@app.callback(
    Output('sir', 'figure'),
    Input('country_SIR', 'value'))
def update_graph_SIR(country_SIR):
    

    global I0,S0,N0,I0,R0,ydata,t,popt

    I0=df_analyse[country_SIR][50]
    S0=N0-I0
    R0=0

    ydata = np.array(df_analyse[country_SIR][50:])
    t=np.arange(len(ydata))

    I0=ydata[0]
    S0=N0-I0
    R0=0



    popt=[0.4,0.1]
    fit_odeint(t, *popt)

    popt, pcov = optimize.curve_fit(fit_odeint, t, ydata)
    perr = np.sqrt(np.diag(pcov))
        
    print('standard deviation errors : ',str(perr), ' start infect:',ydata[0])
    print("Optimal parameters: beta =", popt[0], " and gamma = ", popt[1])

    fitted=fit_odeint(t, *popt)

    fig2=go.Figure()
    title="Fit of SIR model for "+country_SIR+"  cases"
    fig2.add_trace(go.Scatter(x=t, y=ydata, name='Y data', mode='lines'))
    fig2.add_trace(go.Scatter(x=t, y=fitted, name='Fitted data', mode='lines'))
    fig2.update_yaxes(type="log")
    fig2.update_layout(
        title=title,
        xaxis_title="Days",
        yaxis_title="Population Infected"
    )
    return fig2

app.run_server(debug=False, use_reloader=True)
