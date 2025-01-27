import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import warnings
from pandas.plotting import scatter_matrix
import seaborn as sns


### 1) Understanding data issues ###


warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 1.1) Load the data
data = pd.read_csv('Homework 1/dataHW1.csv')

# Check if there are any duplicates
print(sum(data.duplicated()))

# Drop duplicates
data = data.drop_duplicates()

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

# Create the lagged total assets column by shifting 'at' backward by one row (fyear+1 to fyear)
data['lat'] = data.groupby('gvkey')['at'].shift(-1)

# drop the rows where lprice is missing (also drops for lat)
data = data.dropna(subset=['lprice'])

# drop indfmt, consol, popsrc, datafmt, costat, and datadate
data = data.drop(['indfmt', 'consol', 'popsrc', 'datafmt', 'costat', 'datadate'], axis=1)

# check if curcd is the same for all observations
print(data['curcd'].unique()) # Contains 'USD' and 'CAD'

# check if count is the same for all columns
print(data.count())

# print descriptive statistics
print(data.describe())

# Group the data and plot
fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size
data.groupby('fyear')['gvkey'].nunique().plot(
    ax=ax, 
    color='darkblue', 
    linewidth=2, 
    marker='o', 
    markersize=6, 
    linestyle='-'
)

# Adding labels, title, and grid
ax.set_xlabel('Fiscal Year', fontsize=12, labelpad=10, fontweight='bold')
ax.set_ylabel('Number of U.S.-based Companies', fontsize=12, labelpad=10, fontweight='bold')
ax.set_title('Number of Companies Over Time', fontsize=14, fontweight='bold', pad=15)
ax.grid(visible=True, which='major', linestyle='--', linewidth=0.7, alpha=0.7)

# Customize ticks
ax.tick_params(axis='x', labelsize=10, rotation=45) 
ax.tick_params(axis='both', labelsize=10)
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Ensure integer ticks on the x-axis
# Save and show the styled plot
plt.tight_layout()

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


# 1.2) Apply winsorization within each fiscal year for all financial ratios

# Winsorize financial ratios (1st and 99th percentile) in each fiscal year
financial_ratios = [
    'bookleverage1', 'bookleverage2', 'marketvalueofequity', 'marketleverage', 'markettobook', 'assetgrowth',
    'assettangibility', 'roe', 'profitmargin', 'capexratio', 'dividendyield',
    'dividendpayout', 'totalpayout', 'ebitint', 'cash', 'profitability'
]

for ratio in financial_ratios:
    data[f'{ratio}_winsorized'] = data.groupby('fyear')[ratio].transform(lambda x: winsorize(x, lower=0.01, upper=0.99))

# Replace rows with infinite values with NaN
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Generate summary statistics for the winsorized ratios
winsorized_columns = [f'{ratio}_winsorized' for ratio in financial_ratios]
summary_stats12 = data[winsorized_columns].agg(['mean', 'median', 'min', 'max', 'std', 'count']).T
summary_stats12.columns = ['Mean', 'Median', 'Min', 'Max', 'StdDev', 'Non-Missing Count']

# Save the table as a CSV
summary_stats12.to_csv('Winsorized_Statistics12.csv')


# 1.3) Split the firms into 4 quartiles based on the market value of equity
# Assign quartiles to market value of equity within each year
data['market_value_quartile_winsorized'] = data.groupby('fyear')['marketvalueofequity_winsorized'].transform(
    lambda x: pd.qcut(x, 4, labels=[1, 2, 3, 4])  # Assign quartile labels 1-4
)

# Filter for smallest (quartile 1) and largest (quartile 4) groups
filtered_data = data[data['market_value_quartile_winsorized'].isin([1, 4])]

# Calculate summary statistics (mean, median, std) for each variable in financial_ratios
summary_stats13 = filtered_data.groupby('market_value_quartile_winsorized')[financial_ratios].agg(['mean', 'median', 'std'])

# Save the table as a CSV
summary_stats13.to_csv('Quartile_Statistics13.csv')


### 1.4) Financial and non-financial firms ###
# Ensure 'sic' is treated as a string, extract the first two digits, and convert to integer
data['sic_prefix'] = data['sic'].astype(str).str[:2].astype(int)

# Create an indicator for financial firms (SIC code 60-67 inclusive)
data['is_financial'] = data['sic_prefix'].between(60, 67)

# Create an indicator for utility/regulated firms (SIC code 40-49 inclusive)
data['is_utility'] = data['sic_prefix'].between(40, 49)

# Calculate the number of unique financial firms per fiscal year
financial_firms_per_year = (
    data[data['is_financial']].groupby('fyear')['gvkey'].nunique()
)

# Calculate the number of unique non-financial firms per fiscal year
non_financial_firms_per_year = (
    data[~data['is_financial']].groupby('fyear')['gvkey'].nunique()
)

# Combine the results into a DataFrame for each fiscal year
firms_per_year = pd.DataFrame({
    'Financial Firms': financial_firms_per_year,
    'Non-Financial Firms': non_financial_firms_per_year
}).reset_index()

# Save the results to a CSV file
firms_per_year.to_csv('avg_firms_per_year.csv14', index=False)


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
leverage_columns = ['bookleverage1_winsorized', 'bookleverage2_winsorized', 'marketleverage_winsorized']

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

# drop rows with missing values 
data = data.dropna()


### 2) Exploratory data analysis ### 
# focus on the winsorized sample of non-financial and non-utility firms. Focus on six financial ratios: book leverage (1), EBIT interest coverage, cash, profitability, total payout ratio, market-to-book.


# 2.1) a) Histogram-scatter plot matrix
# Subset the data for non-financial and non-utility firms
non_financial_non_utility_firms = data[~data['is_financial'] & ~data['is_utility']]

# Select the six financial ratios
financial_ratios = ['bookleverage1_winsorized', 'ebitint_winsorized', 'cash_winsorized',
                     'profitability_winsorized', 'totalpayout_winsorized', 'markettobook_winsorized'
]

# Subset the data for the selected financial ratios
selected_data = non_financial_non_utility_firms[financial_ratios]

# Create a styled histogram-scatter matrix
fig, axes = plt.subplots(figsize=(12, 12))
scatter_matrix(
    selected_data,
    figsize=(12, 12),
    diagonal="hist",  # Display histograms on the diagonal
    alpha=0.7,  # Transparency for scatter points
    grid=True,  # Enable grid
    color='darkblue',  # Use consistent color
    hist_kwds={'color': 'darkblue', 'edgecolor': 'black', 'alpha': 0.7},  # Style histograms
    marker='o',  # Marker style for scatter
)

# Add a title and adjust the layout
plt.suptitle(
    "Histogram-Scatter Matrix for Selected Financial Ratios (Winsorized Sample)",
    fontsize=14,
    fontweight='bold',
    y=0.98  # Adjust vertical position of the title
)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Avoid overlap with the title
# save graph as png
plt.savefig('Histogram-scatter21a.png')


# 2.1) b) Time-series graph of average and aggregate values of the six financial ratios
# Function to calculate firm-level value
def calculate_firm_value(row):
    """Calculate the firm's value (total assets) for weighting"""
    return row['at']

# Calculate aggregates and averages
results = {}
for ratio in financial_ratios:
    # Group by fiscal year
    yearly_data = non_financial_non_utility_firms.groupby('fyear')
    
    # Calculate simple average (equal-weighted)
    average = yearly_data[ratio].mean()
    
    # Calculate value-weighted average (weighted by firm size)
    weights = non_financial_non_utility_firms.groupby('fyear').apply(
        lambda x: x.apply(calculate_firm_value, axis=1) / x.apply(calculate_firm_value, axis=1).sum()
    )
    value_weighted_avg = (non_financial_non_utility_firms[ratio] * weights.values).groupby(non_financial_non_utility_firms['fyear']).sum()
    
    results[ratio] = {
        'average': average,
        'value_weighted': value_weighted_avg
    }

# Create plots
fig, axes = plt.subplots(len(financial_ratios), 1, figsize=(12, 5 * len(financial_ratios)), sharex=True)

# Ensure axes is always iterable
if len(financial_ratios) == 1:
    axes = [axes]


# Plot averages and value-weighted averages for each financial ratio
for i, ratio in enumerate(financial_ratios):
    ax = axes[i]
    
    # Plot equal-weighted average
    ax.plot(results[ratio]['average'].index, 
            results[ratio]['average'].values, 
            label='Equal-weighted Average', 
            color='blue', 
            marker='o')
    
    # Plot value-weighted average
    ax.plot(results[ratio]['value_weighted'].index, 
            results[ratio]['value_weighted'].values, 
            label='Value-weighted Average', 
            color='orange', 
            linestyle='--', 
            marker='x')
    
    # Customize plot
    ax.set_title(f"Time-Series of {ratio}", fontsize=14, fontweight='bold')
    ax.set_ylabel('Values', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=10)

# Final layout adjustments
axes[-1].set_xlabel('Fiscal Year', fontsize=12)
plt.tight_layout()

# Save the plot
plt.savefig('Time_Series_Financial_Ratios_21b.png')


# 2.2) a) Correlation matrix
# List of winsorized financial ratios

# Original Correlation Matrix
correlation_matrix = non_financial_non_utility_firms[financial_ratios].corr()

# Create heatmap for original correlations
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, 
            annot=True,  # Show correlation values
            cmap='RdBu',  # Red-Blue colormap
            center=0,     # Center the colormap at 0
            vmin=-1,     # Minimum correlation value
            vmax=1,      # Maximum correlation value
            fmt='.2f')   # Format numbers to 2 decimal places

plt.title('Correlation Matrix of Original Financial Ratios', fontsize=14, pad=20, fontweight='bold')
plt.tight_layout()
plt.savefig('Original_Correlation_Matrix22.png')
plt.close()

# Demeaned Variables
# Calculate firm-specific means and subtract them
demeaned_data = pd.DataFrame()
for ratio in financial_ratios:
    firm_means = non_financial_non_utility_firms.groupby('gvkey')[ratio].transform('mean')
    demeaned_data[f'{ratio}_demeaned'] = non_financial_non_utility_firms[ratio] - firm_means

# Calculate correlation matrix for demeaned variables
demeaned_correlation_matrix = demeaned_data.corr()

# Create heatmap for demeaned correlations
plt.figure(figsize=(10, 8))
sns.heatmap(demeaned_correlation_matrix, 
            annot=True,  # Show correlation values
            cmap='RdBu',  # Red-Blue colormap
            center=0,     # Center the colormap at 0
            vmin=-1,     # Minimum correlation value
            vmax=1,      # Maximum correlation value
            fmt='.2f')   # Format numbers to 2 decimal places

plt.title('Correlation Matrix of Demeaned Financial Ratios', fontsize=14, pad=20, fontweight='bold')
plt.tight_layout()
plt.savefig('Demeaned_Correlation_Matrix22.png')
plt.close()

# Save correlation matrices to CSV files
correlation_matrix.to_csv('Original_Correlation_Matrix22.csv')
demeaned_correlation_matrix.to_csv('Demeaned_Correlation_Matrix22.csv')



# 2.3) 
# a) Original OLS
y = non_financial_non_utility_firms['bookleverage1_winsorized']  # Use winsorized variables
x = non_financial_non_utility_firms['profitability_winsorized']
x = sm.add_constant(x)

model1 = sm.OLS(y, x).fit()
model1_robust = sm.OLS(y, x).fit(cov_type='HC1')

model1.summary2().tables[1].to_csv("ols_results23a.csv")
model1_robust.summary2().tables[1].to_csv("ols_results_23a_robust.csv")

# b) Demeaned OLS
y_demeaned = y - non_financial_non_utility_firms.groupby('gvkey')['bookleverage1_winsorized'].transform('mean')
x_demeaned = x['profitability_winsorized'] - non_financial_non_utility_firms.groupby('gvkey')['profitability_winsorized'].transform('mean')
x_demeaned = sm.add_constant(x_demeaned)

model2 = sm.OLS(y_demeaned, x_demeaned).fit()
model2_robust = sm.OLS(y_demeaned, x_demeaned).fit(cov_type='HC1')

model2.summary2().tables[1].to_csv("ols_results_23b.csv")
model2_robust.summary2().tables[1].to_csv("ols_results_23b_robust.csv")

# c) First-difference OLS
non_financial_non_utility_firms = non_financial_non_utility_firms.sort_values(['gvkey', 'fyear'])
y_diff = non_financial_non_utility_firms.groupby('gvkey')['bookleverage1_winsorized'].diff()
x_diff = non_financial_non_utility_firms.groupby('gvkey')['profitability_winsorized'].diff()
x_diff = sm.add_constant(x_diff)

diff_model = sm.OLS(y_diff.dropna(), x_diff.dropna()).fit()
diff_model_robust = sm.OLS(y_diff.dropna(), x_diff.dropna()).fit(cov_type='HC1')

diff_model.summary2().tables[1].to_csv("ols_diff23c.csv")
diff_model_robust.summary2().tables[1].to_csv("ols_diff23c_robust.csv")

# d) Perform firm-by-firm regression for each group
# Initialize lists to store results
firm_results = []
firm_results_robust = []

# Perform firm-by-firm regression for each group if the firm has at least 10 non-missing observations
for gvkey, group in data.groupby('gvkey'):
    # Check for at least 10 non-missing observations
    if group['bookleverage1_winsorized'].notnull().sum() >= 10:
        y_firm = group['bookleverage1'].dropna()
        x_firm = sm.add_constant(group['profitability_winsorized'].loc[y_firm.index])  # Align x and y indices

        # Perform regular OLS regression
        try:
            firm_model = sm.OLS(y_firm, x_firm).fit()
            firm_results.append(firm_model.params[1])  # β_i
        except Exception as e:
            print(f"Error with firm {gvkey} in OLS regression: {e}")

        # Perform OLS regression with robust standard errors
        try:
            firm_model_robust = sm.OLS(y_firm, x_firm).fit(cov_type='HC1')  # Robust standard errors
            firm_results_robust.append(firm_model_robust.params[1])  # β_i
        except Exception as e:
            print(f"Error with firm {gvkey} in robust regression: {e}")

# Summary statistics for β_i
firm_results_df = pd.Series(firm_results).describe().loc[['mean', '50%', 'min', 'max']].rename({'50%': 'median'})
firm_results_df.to_csv("beta23d.csv")

# Summary statistics for β_i with robust standard errors
firm_results_df_robust = pd.Series(firm_results_robust).describe().loc[['mean', '50%', 'min', 'max']].rename({'50%': 'median'})
firm_results_df_robust.to_csv("beta23d_robust.csv")

# Histogram of β_i
plt.figure(figsize=(10, 6))
plt.hist(firm_results, bins=30, color='darkblue', edgecolor='black', alpha=0.7)
plt.title('Histogram of Firm-Specific Coefficients (β_i)', fontsize=14, fontweight='bold')
plt.xlabel('Firm-Specific Coefficients (β_i)', fontweight='bold', fontsize=12, labelpad=10)
plt.ylabel('Frequency', fontsize=12, fontweight='bold', labelpad=10)
plt.grid(axis='y', linestyle='--', linewidth=0.7)
plt.tight_layout()
plt.savefig('Coefficients23d.png')
plt.close()

# Histogram of β_i with robust standard errors
plt.figure(figsize=(10, 6))
plt.hist(firm_results_robust, bins=30, color='darkblue', edgecolor='black', alpha=0.7)
plt.title('Histogram of Firm-Specific Coefficients (β_i) with Robust Standard Errors', fontsize=14, fontweight='bold')
plt.xlabel('Firm-Specific Coefficients (β_i)', fontweight='bold', fontsize=12, labelpad=10)
plt.ylabel('Frequency', fontsize=12, fontweight='bold', labelpad=10)
plt.grid(axis='y', linestyle='--', linewidth=0.7)
plt.tight_layout()
plt.savefig('Coefficients_with_Robust_Standard_Errors23d.png')
plt.close()


# e) Grouping firms by market value

group_results = []
group_results_robust = []

# Perform firm-by-firm regression for each group
for quartile in range(1, 5):
    quartile_data = data[data['market_value_quartile_winsorized'] == quartile]
    quartile_firm_results = []
    
    for gvkey, group in quartile_data.groupby('gvkey'):
        if len(group) >= 10:
            y_firm = group['bookleverage1_winsorized']
            x_firm = sm.add_constant(group['profitability_winsorized'])
            
            try:
                firm_model = sm.OLS(y_firm, x_firm).fit()
                quartile_firm_results.append(firm_model.params[1])  # β_i
            except Exception as e:
                print(f"Error with firm {gvkey}: {e}")
    
    # Summary statistics for each quartile
    quartile_results = pd.Series(quartile_firm_results).describe().loc[['mean', '50%', 'min', 'max']].rename({'50%': 'median'})
    quartile_results['quartile'] = quartile
    group_results.append(quartile_results)

# Combine results for all quartiles
group_results_df = pd.DataFrame(group_results)

# Save results
group_results_df.to_csv("quartile_beta23e.csv", index=False)

# Perform firm-by-firm regression for each group with robust standard errors
for quartile in range(1, 5):
    quartile_data = data[data['market_value_quartile_winsorized'] == quartile]
    quartile_firm_results = []
    
    for gvkey, group in quartile_data.groupby('gvkey'):
        if len(group) >= 10:
            y_firm = group['bookleverage1_winsorized']
            x_firm = sm.add_constant(group['profitability_winsorized'])
            
            try:
                firm_model = sm.OLS(y_firm, x_firm).fit(cov_type='HC1')  # Robust standard errors
                quartile_firm_results.append(firm_model.params[1])  # β_i
            except Exception as e:
                print(f"Error with firm {gvkey}: {e}")
    
    # Summary statistics for each quartile
    quartile_results = pd.Series(quartile_firm_results).describe().iloc[[1, 5, 3, 7]].rename({'50%': 'median'})
    quartile_results['quartile'] = quartile
    group_results_robust.append(quartile_results)

# Combine results for all quartiles
group_results_df_robust = pd.DataFrame(group_results_robust)

# Save results
group_results_df_robust.to_csv("quartile_beta23e_robust.csv", index=False)