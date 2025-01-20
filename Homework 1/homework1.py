import pandas as pd
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


warnings.filterwarnings("ignore", category=RuntimeWarning)

# Load the data
data = pd.read_csv('Homework 1/dataHW1.csv')

# Check if there are any duplicates
print(sum(data.duplicated()))

# Drop duplicates
data = data.drop_duplicates()

# Drop missing values PROBLEM HERE (should i drop all rows with missing values?)
data = data.dropna()

# Check if there are any missing values
print(data.isnull().sum())

# Check the data types
print(data.dtypes)

# Convert the data types
data['datadate'] = pd.to_datetime(data['datadate'])

# Check the data types
print(data.dtypes)

# Only keep US companies
data = data[data['loc'] == 'USA']

# Check if loc only contains USA
print(data['loc'].unique())

# Drop loc
data = data.drop('loc', axis=1)

# Sort the data by gvkey and fyear to ensure proper chronological order
data = data.sort_values(by=['gvkey', 'fyear']).reset_index(drop=True)

# Create the lagged price column by shifting 'prcc_f' backward by one row (fyear+1 to fyear)
data['lprice'] = data.groupby('gvkey')['prcc_f'].shift(-1)

# drop the rows where lprice is missing
data = data.dropna(subset=['lprice'])

# Create the lagged total assets column by shifting 'at' backward by one row (fyear+1 to fyear)
data['lat'] = data.groupby('gvkey')['at'].shift(-1)

# drop the rows where lat is missing
data = data.dropna(subset=['lat'])

# drop indfmt, consol, popsrc, datafmt, costat, and datadate
data = data.drop(['indfmt', 'consol', 'popsrc', 'datafmt', 'costat', 'datadate'], axis=1)

# check if curcd is the same for all observations
print(data['curcd'].unique()) # Contains 'USD' and 'CAD'

# check if count is the same for all columns
print(data.count())

# print descriptive statistics
print(data.describe())

# Plot the evolution in the number of companies over time
data.groupby('fyear')['gvkey'].nunique().plot()
plt.xlabel('Year')
plt.ylabel('Number of companies')
plt.title('Number of companies over time')
plt.show()

# save graph as png
plt.savefig('Number of companies over time.png')




# Define a function to winsorize data
def winsorize(series, lower=0.01, upper=0.99):
    lower_bound = series.quantile(lower)
    upper_bound = series.quantile(upper)
    return series.clip(lower=lower_bound, upper=upper_bound)

# Financial ratios
data['bookleverage1'] = (data['dlc'] + data['dltt']) / data['at']
data['bookleverage2'] = data['lt'] / data['at']
data['marketvalueofequity'] = data['csho'] * data['prcc_f']
data['marketleverage'] = data['bookleverage1'] / (data['bookleverage1'] + data['pstk'] + data['marketvalueofequity'])
data['markettonook'] = (data['prcc_f'] * data['csho'] + data['dltt'] + data['dlc'] + data['pstkl'] - data['txditc']) / data['at']
data['assetgrowth'] = data['at'] / data['lat'] - 1
data['assettangibility'] = data['ppent'] / data['at']
data['roe'] = data['ni'] / data['ceq']
data['profitmargin'] = data['ni'] / data['sale']
data['capexratio'] = data['capx'] / data['at']
data['dividendyield'] = (data['dv'] / data['csho']) / data['lprice']
data['dividendpayout'] = data['dv'] / data['ni']
data['totalpayout'] = (data['dv'] + data['prstkc']) / data['ni']
data['ebitint'] = data['ebit'] / data['xint']
data['cash'] = data['che'] / data['at']
data['profitability'] = data['oibdp'] / data['at']


print(data.isnull().sum())  # Count of NaN values


# Winsorize financial ratios (1st and 99th percentile) in each fiscal year
financial_ratios = [
    'bookleverage1', 'bookleverage2', 'marketleverage', 'markettonook', 'assetgrowth',
    'assettangibility', 'roe', 'profitmargin', 'capexratio', 'dividendyield',
    'dividendpayout', 'totalpayout', 'ebitint', 'cash', 'profitability'
]

# Apply winsorization within each fiscal year for all financial ratios
for ratio in financial_ratios:
    data[ratio] = data.groupby('fyear')[ratio].transform(lambda x: winsorize(x, lower=0.01, upper=0.99))

### Remove rows with infinite values??? ###
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

summary_stats = data[financial_ratios].agg(['mean', 'median', 'min', 'max', 'std', 'count']).T
summary_stats.columns = ['Mean', 'Median', 'Min', 'Max', 'StdDev', 'Non-Missing Count']
print(summary_stats)

