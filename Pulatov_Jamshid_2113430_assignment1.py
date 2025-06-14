# Assigment 1 by FinTech group
# Group members:
# Egamberdiev Temurbek
# Pulatov Jamshid
# Abdurayimov Jalol
# Ruzimurodov Abbos


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from numpy.linalg import solve

#Loading the dataset
df = pd.read_csv('https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/current.csv?sc_lang=en&hash=80445D12401C59CF716410F3F7863B64')

#clean the DataFrame by removing the row with transformation codes
df_cleaned = df.drop(index=0)
df_cleaned.reset_index(drop=True, inplace=True)
df_cleaned['sasdate'] = pd.to_datetime(df_cleaned['sasdate'], format='%m/%d/%Y')
df_cleaned


##showing result in terminal
##print(df_clened)

#Extract transformation codes
transformation_codes = df.iloc[0,1:].to_frame().reset_index()
transformation_codes.columns = ["Series", 'Transformation_Code']

#Function to apply transformations based on the transformation code

def apply_transformation(series, code):
    if code == 1:
        # No transformation
        return series
    elif code == 2:
        # First difference
        return series.diff()
    elif code == 3:
        # Second difference
        return series.diff().diff()
    elif code == 4:
        # Log
        return np.log(series)
    elif code == 5:
        # First difference of log
        return np.log(series).diff()
    elif code == 6:
        # Second difference of log
        return np.log(series).diff().diff()
    elif code == 7:
        # Delta (x_t/x_{t-1} - 1)
        return series.pct_change()
    else:
        raise ValueError("Invalid transformation code")


# Applying  the transformation to each column in df_cleaned based on transform
for series_name, code in transformation_codes.values:
    df_cleaned[series_name] = apply_transformation(df_cleaned[series_name].astype(float), float(code))

df_cleaned = df_cleaned[2:]
df_cleaned.reset_index(drop=True, inplace=True)
df_cleaned.head()


series_to_plot = ['INDPRO', 'CPIAUCSL', 'TB3MS']
series_names = ['Industrial Production',
                'Inflation (CPI)', 
                '3-month Treasury Bill rate']

#Create a figure and a grid of subplots
fig, axs = plt.subplots(len(series_to_plot), 1, figsize = (8, 15))

#Iterate over the selected series and plot each one 
for ax, series_name, plot_title in zip(axs, series_to_plot, series_names):
    if series_name in df_cleaned.columns:
        dates = pd.to_datetime(df_cleaned['sasdate'], format='%m/%d/%Y')
        ax.plot(dates, df_cleaned[series_name], label=plot_title)
        ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_title(plot_title)
        ax.set_xlabel('Year')
        ax.set_ylabel('Transformed Value')
        ax.legend(loc='upper left')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation = 45, ha = 'right')
    else:
        ax.set_visible(False)   # hides plots for which the data is not available

plt.tight_layout()
plt.show()



Yraw = df_cleaned['INDPRO']
Xraw = df_cleaned[['CPIAUCSL', 'TB3MS']]

num_lags = 4 ## this is p
num_leads = 1 ## this is h
X = pd.DataFrame()
## Add the lagged values of Y
col = 'INDPRO'
for lag in range(0, num_lags+1):
    # shift each column in the DataFrame and name it  with a lag suffix
    X[f'{col}_lag{lag}'] = Yraw.shift(lag)

for col in Xraw.columns:
    for lag in range (0, num_lags+1):
        #Shift each column in the DataFrame and name it with lag a suffix
        X[f"{col}_lag{lag}"] = Xraw[col].shift(lag)

## Add a column on ones (for the intercept)
X.insert(0, 'Ones', np.ones(len(X)))


X.head()

# creating Y vector
y = Yraw.shift(-num_leads)
y

## Save last row of X (converted to numpy)
X_T = X.iloc[-1:].values
## Subset getting only rows of X and y from p+1 to h-1
## and convert to numpy array
y = y.iloc[num_lags:-num_leads].values
X = X.iloc[num_lags:-num_leads].values

print(X_T)

# solving for the OLS estimator beta: (X^X)^{-1} X'Y
beta_ols = solve(X.T @ X, X.T @y)

# produce the one step ahead forecast
# % change month-to-month INDPRO
forecast = X_T@beta_ols*100
forecast

def calculate_forecast(df_cleaned, p=4, H=[1,4,8], end_date='12/1/1999', target='INDPRO', xvars=['CPIAUCSL', 'TB3MS']):

    # subset df_cleaned to use only data up to end_date
    rt_df = df_cleaned[df_cleaned['sasdate'] <= pd.Timestamp(end_date)]
    ## get the actual values of target at different steps ahead
    Y_actual = []
    for h in H:
        os = pd.Timestamp(end_date) + pd.DateOffset(months=h)
        Y_actual.append(df_cleaned[df_cleaned['sasdate'] == os][target]*100)
        # now Y contains the true values at T+H (multiplying * 100)


    Yraw = rt_df[target]
    Xraw = rt_df[xvars]

    X = pd.DataFrame()
    ## add the lagged values of Y
    for lag in range(0, p):
        # shift each column in the DataFrame and name it with a lag suffix
        X[f'{target}_lag{lag}'] = Yraw.shift(lag)

    for col in Xraw.columns:
        for lag in range(0, p):
            X[f'{col}_lag{lag}'] = Xraw[col].shift(lag)


    #add a column on ones (for the intercept)
    X.insert(0, 'Ones', np.ones(len(X)))

    # save last row of X (converted to numpy)
    X_T = X.iloc[-1:].values

    # while the X will be the same, Y needs to be leaded differently
    Yhat = []
    for h in H:
        y_h = Yraw.shift(-h)
        ## subset getting only rows of X and y from p+1 to h-1
        y = y_h.iloc[p:-h].values
        X_ = X.iloc[p:-h].values
        # solving for the OLS estimator beta: (X'X)^{-1} X'Y
        beta_ols = solve(X_.T @ X_, X_.T @ y)
        ## produce the one step ahead forecast
        ## %change month-to-month INDPR
        Yhat.append(X_T@beta_ols*100)

    #now calculate the forecasting error and return

    return np.array(Y_actual) - np.array(Yhat)

t0 = pd.Timestamp('12/1/1999')
e = []
T = []
for j in range(0,10):
    t0 = t0 + pd.DateOffset(months=1)
    print(f'Using data up to {t0}')
    ehat = calculate_forecast(df_cleaned, p=4, H=[1,4,8], end_date=t0)
    e.append(ehat.flatten())
    T.append(t0)

# create a pandas DataFrame from the list
edf = pd.DataFrame(e)
# calculate the RMSFE, that is, the square root of the MSFE
print(np.sqrt(edf.apply(np.square).mean()))