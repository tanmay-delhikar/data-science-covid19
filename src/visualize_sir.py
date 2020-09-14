import pandas as pd
import numpy as np

from datetime import datetime
import pandas as pd 

import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns
from scipy import optimize
from scipy import integrate

sns.set(style="darkgrid")

mpl.rcParams['figure.figsize'] = (16, 9)
pd.set_option('display.max_rows', 500)

from  build_features  import *
from  process_data import *


proces_SIR_data()

df_analyse=pd.read_csv('../data/processed/COVID_small_flat_table.csv',sep=';')  
df_analyse.sort_values('date',ascending=True).head()

N0=1000000 #max susceptible population
beta=0.4   # infection spread dynamics
gamma=0.1  # recovery rate


# condition I0+S0+R0=N0
I0=df_analyse.Germany[35]
S0=N0-I0
R0=0


ydata = np.array(df_analyse.Germany[35:])
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
    dS_dt=-beta*S*I/N0          #S*I is the 
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

# plt.semilogy(t, ydata, 'o')
# plt.semilogy(t, fitted)
# plt.title("Fit of SIR model for Germany cases")
# plt.ylabel("Population infected")
# plt.xlabel("Days")
# plt.show()
# print("Optimal parameters: beta =", popt[0], " and gamma = ", popt[1])
# print("Basic Reproduction Number R0 " , popt[0]/ popt[1])
# print("This ratio is derived as the expected number of new infections (these new infections are sometimes called secondary infections from a single infection in a population where all subjects are susceptible. @wiki")




# import plotly.express as px
# df = px.data.gapminder().query("continent=='Oceania'")
# print(df)
# fig = px.scatter(x=t, y=[ydata,fitted],log_y=True)
# fig.show()