import pandas as pd
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pandas.plotting import scatter_matrix


### 1) Understanding data issues ###


warnings.filterwarnings("ignore", category=RuntimeWarning)

# 1.1) Load the data
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
plt.savefig('Number of companies over time11.png')

### 1.2) Financial ratios ###
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
data['markettobook'] = (data['prcc_f'] * data['csho'] + data['dltt'] + data['dlc'] + data['pstkl'] - data['txditc']) / data['at']
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

# 1.2) Apply winsorization within each fiscal year for all financial ratios

# Winsorize financial ratios (1st and 99th percentile) in each fiscal year
financial_ratios = [
    'bookleverage1', 'bookleverage2', 'marketleverage', 'markettobook', 'assetgrowth',
    'assettangibility', 'roe', 'profitmargin', 'capexratio', 'dividendyield',
    'dividendpayout', 'totalpayout', 'ebitint', 'cash', 'profitability'
]

for ratio in financial_ratios:
    data[ratio] = data.groupby('fyear')[ratio].transform(lambda x: winsorize(x, lower=0.01, upper=0.99))

# Remove rows with infinite values
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

summary_stats12 = data[financial_ratios].agg(['mean', 'median', 'min', 'max', 'std', 'count']).T
summary_stats12.columns = ['Mean', 'Median', 'Min', 'Max', 'StdDev', 'Non-Missing Count']

# Display the summary table
print(summary_stats12)

# Save the table as a CSV
summary_stats12.to_csv('Winsorized_Statistics12.csv')

### 1.3) Split the firms into 4 quartiles based on the market value of equity ###
data['market_value_quartile'] = data.groupby('fyear')['marketvalueofequity'].transform(
    lambda x: pd.qcut(x, 4, labels=False)
)

# Filter for smallest (quartile 0) and largest (quartile 3) groups
filtered_data = data[data['market_value_quartile'].isin([0, 3])]

# Calculate summary statistics for the selected quartiles
quartile_stats13 = filtered_data.groupby(['fyear', 'market_value_quartile'])[financial_ratios].agg(['mean', 'median', 'std'])

# Display the summary table
print(quartile_stats13)

# Save the table as a CSV
quartile_stats13.to_csv('Quartile_Statistics13.csv')

### 1.4) Financial and non-financial firms ###
# Create an indicator for financial firms (SIC code 60-67 inclusive)
data['is_financial'] = data['sic'].astype(str).str[:2].astype(int).between(60, 67)

# Create an indicator for utility/regulated firms (SIC code 40-49 inclusive)
data['is_utility'] = data['sic'].astype(str).str[:2].astype(int).between(40, 49)

# Check the unique values for the indicators
print(data[['is_financial', 'is_utility']].sum())

# Calculate the average number of financial and non-financial firms per fiscal year
financial_counts = data.groupby(['fyear', 'is_financial'])['gvkey'].nunique().reset_index()
financial_counts = financial_counts.pivot(index='fyear', columns='is_financial', values='gvkey')
financial_counts.columns = ['Non-Financial Firms', 'Financial Firms']
financial_counts.fillna(0, inplace=True)

# Calculate the average number of firms across years
avg_financial_counts = financial_counts.mean()
print("Average number of firms per fiscal year:")
print(avg_financial_counts)

# Calculate the number of utility firms per fiscal year
utility_counts14 = data.groupby('fyear')['is_utility'].sum()

# Display the summary table
print(utility_counts14)

# Save the table as a CSV
utility_counts14.to_csv('Industries_Statistics14.csv')

# Visualize the trend in financial vs non-financial firms over time
financial_counts.plot(kind='bar', figsize=(12, 6))
plt.title('Number of Financial vs Non-Financial Firms Over Time')
plt.xlabel('Fiscal Year')
plt.ylabel('Number of Firms')
plt.legend(title='Firm Type')
# Update the x-axis tick labels to remove decimal points
plt.xticks(ticks=plt.xticks()[0], labels=financial_counts.index.astype(int))
plt.tight_layout()
# save graph as png
plt.savefig('Number of Financial vs Non-Financial Firms Over Time14.png')

### 1.5) Using the book leverages and market leverage, create descritive statistics for the subset of firms ###
# Define subsets
financial_firms = data[data['is_financial']]
utility_firms = data[data['is_utility']]
non_financial_non_utility_firms = data[~data['is_financial'] & ~data['is_utility']]

# Filter for non-financial, non-utility firms with non-missing values of total assets throughout the sample period
non_financial_complete_assets = (
    non_financial_non_utility_firms.groupby('gvkey')['at']
    .transform(lambda x: x.notnull().all())
)
non_financial_complete_assets_firms = non_financial_non_utility_firms[non_financial_complete_assets]

# Define function to calculate summary statistics
def calculate_statistics(df, columns):
    return df[columns].agg(['mean', 'median', 'std', 'count']).T

# List of leverage columns
leverage_columns = ['bookleverage1', 'bookleverage2', 'marketleverage']

# Calculate statistics for each group
financial_stats = calculate_statistics(financial_firms, leverage_columns)
utility_stats = calculate_statistics(utility_firms, leverage_columns)
non_financial_non_utility_stats = calculate_statistics(non_financial_non_utility_firms, leverage_columns)
non_financial_complete_assets_stats = calculate_statistics(non_financial_complete_assets_firms, leverage_columns)

# Combine results into a single table
summary_table15 = pd.concat(
    {
        "Financial Firms": financial_stats,
        "Utility Firms": utility_stats,
        "Non-Financial & Non-Utility Firms": non_financial_non_utility_stats,
        "Non-Financial Complete Assets Firms": non_financial_complete_assets_stats,
    },
    axis=1,
)

# Display the summary table
print(summary_table15)

# Save the table as a CSV
summary_table15.to_csv('Leverage_Statistics15.csv')



### 2) Exploratory data analysis ### 
# focus on the winsorized sample of non-financial and non-utility firms. Focus on six financial ratios: book leverage (1), EBIT interest coverage, cash, profitability, total payout ratio, market-to-book.


# 2.1) a) Histogram-scatter plot matrix
# Subset the data for non-financial and non-utility firms
non_financial_non_utility_firms = data[~data['is_financial'] & ~data['is_utility']]

# Select the six financial ratios
financial_ratios = ['bookleverage1', 'ebitint', 'cash', 'profitability', 'totalpayout', 'markettobook']

# Subset the data for the selected financial ratios
selected_data = non_financial_non_utility_firms[financial_ratios]

# Create a histogram-scatter matrix
scatter_matrix(
    selected_data,
    figsize=(12, 12),
    diagonal="hist",  # Display histograms on the diagonal
    alpha=0.7,  # Transparency for scatter points
    grid=True,  # Enable grid
)

# Adjust layout and show the plot
plt.suptitle("Histogram-Scatter Matrix for Selected Financial Ratios (Winsorized Sample)", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Avoid overlap with the title
# save graph as png
plt.savefig('Histogram-scatter21a.png')


# 2.1) a) Time-series graph of average and aggregate values of the six financial ratios
# Subset the data for non-financial and non-utility firms
non_financial_non_utility_firms = data[~data['is_financial'] & ~data['is_utility']]

# Define financial ratios and their components (numerator/denominator for aggregate calculation)
ratios_components = {
    'bookleverage1': ('dlc + dltt', 'at'),  # Book leverage 1: (DLC + DLTT) / AT
    'ebitint': ('ebit', 'xint'),           # EBIT interest coverage: EBIT / XINT
    'cash': ('che', 'at'),                 # Cash: CHE / AT
    'profitability': ('oibdp', 'at'),      # Profitability: OIBDP / AT
    'totalpayout': ('dv + prstkc', 'ni'),  # Total payout: (DV + PRSTKC) / NI
    'markettobook': ('prcc_f * csho + dltt + dlc + pstkl - txditc', 'at')  # Market-to-book
}

# Initialize a dictionary to store aggregated data
aggregated_results = {}

# Create a figure with subplots (dynamic size based on the number of ratios)
num_ratios = len(ratios_components)
fig, axes = plt.subplots(num_ratios, 1, figsize=(10, 5 * num_ratios))  # One row per ratio

# If there's only one ratio, ensure axes is iterable
if num_ratios == 1:
    axes = [axes]

# Make a deep copy of the DataFrame to avoid the SettingWithCopyWarning
df = non_financial_non_utility_firms.copy()

# Loop through each ratio and calculate average and aggregate values over time
for idx, (ratio, (numerator, denominator)) in enumerate(ratios_components.items()):
    # Ensure explicit use of .loc to avoid warnings
    df.loc[:, numerator] = df.eval(numerator)
    df.loc[:, denominator] = df.eval(denominator)

    # Calculate average values (mean)
    average_values = df.groupby('fyear')[ratio].mean()

    # Calculate aggregate values (sum of numerator / sum of denominator)
    aggregate_values = (
        df.groupby('fyear')[numerator].sum() /
        df.groupby('fyear')[denominator].sum()
    )

    # Plot on the corresponding axis
    ax = axes[idx]
    ax.plot(average_values.index, average_values, label='Average')
    ax.plot(aggregate_values.index, aggregate_values, label='Aggregate')
    ax.set_title(f'Time Series for {ratio}')
    ax.set_xlabel('Fiscal Year')
    ax.set_ylabel(ratio)
    ax.legend()
    ax.grid()

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the entire figure as one image
plt.savefig('Timeseries_graphs_combined21b.png')


# 2.2) Correlation matrix
# Calculate the correlation matrix for the selected financial ratios
correlation_matrix22a = selected_data.corr()
# Save the table as a CSV
correlation_matrix22a.to_csv('Correlation_Matrix22a.csv')

# Tranform each of the selected financial ratios by removing the firm-specific mean and create a new correlation matrix
selected_data_centered = selected_data.sub(selected_data.mean())
correlation_matrix_centered = selected_data_centered.corr()
# Save the table as a CSV
correlation_matrix_centered.to_csv('Correlation_Matrix_Centered22b.csv')

