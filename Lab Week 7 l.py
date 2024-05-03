# -*- coding: utf-8 -*-
"""
Created on Thu May  2 19:42:13 2024

@author: ChelseySSS
"""

#SHIHAN ZHAO

import pandas as pd
import numpy as np 
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression 

url_to_csv = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv'
df = pd.read_csv(url_to_csv)


# 1) Explore the data & produce some basic summary stats  

# Display the first few rows of the dataframe
print("First few rows of the dataframe:")
print(df.head())

# Check the data types of each column
print("\nData types of each column:")
print(df.dtypes)

# Check for missing values
print("\nCheck for missing values:")
print(df.isnull().sum())

# Generate descriptive statistics
print("\nDescriptive statistics:")
print(df.describe())

# Optionally, additional information such as value counts for categorical columns
print("\nValue counts for the 'cut' column:")
print(df['cut'].value_counts())

print("\nValue counts for the 'color' column:")
print(df['color'].value_counts())

print("\nValue counts for the 'clarity' column:")
print(df['clarity'].value_counts())




# 2) Run a regression of price (y) on carat (x), including an 
#    intercept term.  Report the estimates of the intercept & slope 
#    coefficients using each of the following methods:
#        a) NumPy
#        b) statsmodels (smf) 
#        c) statsmodels (sm)
#        d) scikit-learn (LinearRegression)  
#           Hint:  scikit-learn only works with array-like objects.    
#    Confirm that all four methods produce the same estimates.

# Extracting X and Y variables
X = df['carat'].values
Y = df['price'].values

# a) NumPy
# Adding a column of ones to X to include an intercept in the model
X_np = np.c_[np.ones(X.shape[0]), X]
beta_numpy = np.linalg.inv(X_np.T.dot(X_np)).dot(X_np.T).dot(Y)
print("NumPy estimates:")
print(f"Intercept: {beta_numpy[0]}, Slope: {beta_numpy[1]}")

# b) statsmodels (smf)
model_smf = smf.ols('price ~ carat', data=df).fit()
print("\nStatsmodels Formula estimates:")
print(f"Intercept: {model_smf.params['Intercept']}, Slope: {model_smf.params['carat']}")

# c) statsmodels (sm)
# adding a constant for intercept
X_sm = sm.add_constant(df['carat'])  
model_sm = sm.OLS(Y, X_sm).fit()
print("\nStatsmodels OLS estimates:")
print(f"Intercept: {model_sm.params['const']}, Slope: {model_sm.params['carat']}")

# d) scikit-learn (LinearRegression)
# scikit-learn expects X to be a 2D array
X_sklearn = X.reshape(-1, 1)
model_sklearn = LinearRegression().fit(X_sklearn, Y)
print("\nScikit-Learn estimates:")
print(f"Intercept: {model_sklearn.intercept_}, Slope: {model_sklearn.coef_[0]}")





# 3) Run a regression of price (y) on carat, the natual logarithm of depth  
#    (log(depth)), and a quadratic polynomial of table (i.e., include table & 
#    table**2 as regressors).  Estimate the model parameters using any Python
#    method you choose, and display the estimates.  

# Since we need the natural logarithm of the depth, let's add that to the DataFrame
df['log_depth'] = np.log(df['depth'])

# Adding a column for table squared
df['table_squared'] = df['table'] ** 2

# Define the regression model
model = smf.ols('price ~ carat + log_depth + table + table_squared', data=df).fit()

# Display the model's coefficients
print("Model parameters:")
print(model.params)

# Display the summary of the model to see more details like R-squared, p-values, etc.
print("\nModel summary:")
print(model.summary())




# 4) Run a regression of price (y) on carat and cut.  Estimate the model 
#    parameters using any Python method you choose, and display the estimates.  

# Define the regression model including 'cut' as a categorical variable
model = smf.ols('price ~ carat + C(cut)', data=df).fit()

# Display the model's coefficients
print("Model parameters:")
print(model.params)

# Display the summary of the model to see more details like R-squared, p-values, etc.
print("\nModel summary:")
print(model.summary())




# 5) Run a regression of price (y) on whatever predictors (and functions of 
#    those predictors you want).  Try to find the specification with the best
#    fit (as measured by the largest R-squared).  Note that this type of data
#    mining is econometric blasphemy, but is the foundation of machine
#    learning.  Fit the model using any Python method you choose, and display 
#    only the R-squared from each model.  We'll see who can come up with the 
#    best fit by the end of lab.  
 
# Define a list of model specifications to test
models = [
    'price ~ carat',
    'price ~ carat + C(cut)',
    'price ~ carat + C(cut) + C(color)',
    'price ~ carat + C(cut) + C(color) + C(clarity)',
    'price ~ carat + carat**2 + C(cut)',
    'price ~ carat + np.log(carat) + C(cut) + C(color)',
    'price ~ carat + C(cut) + C(color) + carat:C(color)',
    'price ~ carat + C(cut) + C(color) + carat:C(cut) + C(clarity)',
    'price ~ carat + np.log(carat) + C(cut) + C(color) + C(clarity) + carat:C(color) + carat:C(clarity)',
    'price ~ carat + carat**2 + C(cut) + C(color) + C(clarity) + carat:C(color) + carat:C(cut)'
]

# Evaluate each model and keep track of R-squared
best_r_squared = 0
best_model = None

for model_spec in models:
    model = smf.ols(model_spec, data=df).fit()
    if model.rsquared > best_r_squared:
        best_r_squared = model.rsquared
        best_model = model_spec

    print(f"Model: {model_spec}\nR-squared: {model.rsquared}\n")

# Print the best model's R-squared
print(f"Best model specification: {best_model}")
print(f"Best R-squared: {best_r_squared}")    

    

    
    
  
