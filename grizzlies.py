# Author: Eq6 Bedu
# Date: 08.january.21
# Description: This module contains functions to use in Data Science
# Grizzlies Utilities Module
# Project: Bedu Data Science

# Description: This module contains functions to use in Data Science

# Libraries:
import requests
import json 
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------
# ETL Functions
# ---------------------------------------------------------------

# Function to load data from an API
# Returns a Normalized DataFrame
# Parameters:
#   url: String with the url to get the data
#   normalize: Boolean to normalize the data
#   elements: Integer with the number of elements to show
def load_data_api(url, normalize = False, elements = 5):
    try: 
        response = requests.get(url)
        if response.status_code == 200:
            data = json.loads(response.content)
            if normalize:
                data = pd.json_normalize(data)
                print('Data normalized')
            df = pd.DataFrame(data)
            print('Dataframe Info')
            print('-----------------------------------------------------------------------------')
            print(df.head(elements))
            print('-----------------------------------------------------------------------------')
            print(df.info())
            return df
        else:
            raise Exception("Failed to retrieve data from the API")
    except Exception as e:
        print(str(e))
        
# Function to delete a column from a DataFrame
# Returns a DataFrame
# Parameters:
#   df: DataFrame with the data to transform
#   column: String with the column to delete
def delete_column(df, column):
    try:
        df.drop(column, axis = 1, inplace = True)
        print(f'Column {column} deleted')
        print('Dataframe Info')
        print('-----------------------------------------------------------------------------')
        print(df.info())
        return df
    except Exception as e:
        print(str(e))

# Function ETL: Rename columns
# Returns a DataFrame
# Parameters:   
#   df: DataFrame with the data to transform
#   columns: List with the columns to keep
#   new_names: List with the columns to rename
def rename_columns(df, columns, new_names):
    try:
        for column in columns:
            df.rename(columns = {column:new_names[columns.index(column)]}, inplace = True)
            print(f'Column Renamed: {column} to {new_names[columns.index(column)]}')
        print('Dataframe Info')
        print('-----------------------------------------------------------------------------')
        print(df.info())
        return df
    except Exception as e:
        print(str(e))
    
# Function ETL: Recast Columns
# Returns a DataFrame
# Parameters:
#   df: DataFrame with the data to transform
#   columns: List with the columns to transform
#   datatypes: List with the data types to transform
# IF date pd.to_datetime(df['Date'])
def recast_columns(df, columns, datatypes):
    datatype_map = {column: datatype for column, datatype in zip(columns, datatypes)}
    try:
        for column in columns:
            if datatype_map[column] == 'date':
                df[column] = pd.to_datetime(df[column])
            else:
                df[column] = df[column].astype(datatype_map[column])
            print(f'Column Recasted: {column} to {datatype_map[column]}')
        print('Dataframe Info')
        print('-----------------------------------------------------------------------------')
        print(df.info())
        return df
    except Exception as e:
        print(str(e))
        
# Function ETL: Specify index
# Returns a DataFrame
# Parameters:
#   df: DataFrame with the data to transform
#   index: String with the column to use as index
def set_index(df, index):
    try:
        df.set_index(index, inplace = True)
        print(f'Index specified: {index}')
        print('Dataframe Info')
        print('-----------------------------------------------------------------------------')
        print(df.info())
        return df
    except Exception as e:
        print(str(e))
    
# Function to print a detailed report of the DataFrame
# Parameters:
#  df: DataFrame with the data to analyze
def print_report(df):
    print('DATAFRAME REPORT')
    print('-----------------------------------------------------------------------------')
    print(f"DataFrame shape: {df.shape}")
    print('-----------------------------------------------------------------------------')
    print(f"Column names: {df.columns.tolist()}")
    print('-----------------------------------------------------------------------------')
    print(f"Data types:\n{df.dtypes}")
    print('-----------------------------------------------------------------------------')
    print(f"Summary statistics:\n{df.describe()}")

# ---------------------------------------------------------------
# Cleaning Functions
# ---------------------------------------------------------------

# Function to create a integrity report fo ta dataframe
# Checks for na and duplicate values and values not corresponding to the data type
# Parameters:
#  df: DataFrame with the data to analyze
def integrity_report(df):
    print('Integrity Report:')
    print('------------------------------------------------------------------------------')
    print(f'NA values:\n{df.isna().sum()}')
    print('\n------------------------------------------------------------------------------\n')
    print(f'Duplicate values:{df.duplicated().sum()}')
    print('\n------------------------------------------------------------------------------\n')
    print('Unique values per column')
    for column in df.columns:
        print(f'{column}: {df[column].nunique()}')      
    print('\n-----------------------------------------------------------------------------\n')
    print(f'Values not corresponding to the data type:\n{df.apply(lambda x: x.apply(type).value_counts())}')

# Function to check unique values and its frequency from a set of columns
# Parameters:
#   df: DataFrame with the data to transform
#   columns: List with the columns to check
def report_unique(df, columns):
    print('-----------------------------------------------------------------------------')        
    try:
        for column in columns:
            print(f'Unique values for {column}:')
            print(df[column].value_counts())
    except Exception as e:
        print(str(e))

# Function to drop rows with na values
# Parameters:
#   df: DataFrame with the data to transform
#   inplace: Boolean to modify the original DataFrame
def drop_na(df):
    try:
        df.dropna(inplace = True)
        print(f'{df.isna().sum().sum()} NA values dropped')
        return df
    except Exception as e:
        print(str(e))
        
# Function to drop duplicates
# Parameters:
#   df: DataFrame with the data to transform
#   inplace: Boolean to modify the original DataFrame
def drop_duplicates(df):
    try:
        df.drop_duplicates(inplace = True)
        print(f'{df.duplicated().sum()} Duplicates dropped')
        return df
    except Exception as e:
        print(str(e))
           
# Function replace na values with a random value from the mean 
# of the column with a given upper and lower values. 
# limits is calculated as:
# Upper limit: The mean of the column + random value between the mean and given value
# Lower Limit: The mean of the column - random value between the mean and given value
# Parameters:
#   df: DataFrame with the data to transform
#   column: String with the column to modify
#   limits: List with the limits to generate the random value
#   inplace: Boolean to modify the original DataFrame
def replace_na(df, column, upper_value = 0, lower_value = 0):
    try:
        mean = df[column].mean()
        upper_limit = mean + random.uniform(mean, upper_value)
        lower_limit = mean - random.uniform(mean, lower_value)
        df[column].fillna(random.uniform(lower_limit, upper_limit), inplace = True)
        print(f'NA values replaced with random values between {lower_limit} and {upper_limit}')
        return df
    except Exception as e:
        print(str(e))

# ---------------------------------------------------------------
# EDA Functions
# ---------------------------------------------------------------

# Function to get the outliers of a the df
# Parameters:
#   df: DataFrame with the data to transform
#   clean: Boolean to clean the outliers
def check_outliers(df, clean = False):
    try:
        print('Outliers')
        print('-----------------------------------------------------------------------------')
        for column in df.columns:
            if df[column].dtype != 'object':
                print(f'Column: {column}')

                q1 = df[column].quantile(0.25)
                q3 = df[column].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - (1.5 * iqr)
                upper_bound = q3 + (1.5 * iqr)
                print(f'Lower bound: {lower_bound}')
                print(f'Upper bound: {upper_bound}')
                print(f'Outliers: {df[(df[column] < lower_bound) | (df[column] > upper_bound)].shape[0]}')
                if clean:
                    df = df[(df[column] > lower_bound) & (df[column] < upper_bound)]
                print('-----------------------------------------------------------------------------')
        return df
    except Exception as e:
        print(str(e))

# Function to analyze the distribution of categorical columns in a dataframe
# Checks if the columns are categorical and then plots a barplot
# Plots all together in a subplot
# Parameters:
#   df: DataFrame with the data to transform
def categorical_analysis(df):
    try:
        categorical_columns = df.select_dtypes(include = ['object']).columns.tolist()
        print('Categorical Analysis')
        print('-----------------------------------------------------------------------------')
        for column in categorical_columns:
            print(f'Column: {column}')
            print('-----------------------------------------------------------------------------')
            sns.barplot(x = df[column].value_counts().index, y = df[column].value_counts())
            plt.show()
    except Exception as e:
        print(str(e))


# Function to analyze the distribution of numeric columns in a dataframe
# Checks if the columns are numerics and then plots a histogram
# Adds a line with the mean of the column
# Parameters:
#   df: DataFrame with the data to transform
#   bins = Number of bins to use in the histogram
def distribution_analysis(df, bins = 10):
    try:
        numeric_columns = df.select_dtypes(include = ['int64', 'float64']).columns.tolist()
        print('Numeric Analysis')
        print('-----------------------------------------------------------------------------')
        for column in numeric_columns:
            print(f'Column: {column}')
            print('-----------------------------------------------------------------------------')
            sns.histplot(df[column], bins = bins)
            plt.axvline(df[column].mean(), color = 'darkblue', linestyle = 'dashed', linewidth = 0.75)
            plt.show()
    except Exception as e:
        print(str(e))
        

# Function to analyze the correlation of numeric columns in a dataframe
# Checks if the columns are numerics and then calculates the correlation.
# Plots a heatmap and a correlation matrix
# Parameters:
#   df: DataFrame with the data to transform
def correlation_analysis(df):
    try:
        numeric_columns = df.select_dtypes(include = ['int64', 'float64']).columns.tolist()
        corr_matrix = df[numeric_columns].corr()
        print('Correlation Analysis')
        print('-----------------------------------------------------------------------------')
        print('Correlation Matrix')
        print('-----------------------------------------------------------------------------')
        print(corr_matrix)
        print('-----------------------------------------------------------------------------')
        print('Heatmap')
        print('-----------------------------------------------------------------------------')
        mask = np.zeros_like(corr_matrix)
        mask[np.triu_indices_from(mask)] = True
        sns.set(font_scale=0.65)
        sns.heatmap(corr_matrix, annot=True, annot_kws={"size": 7}, mask=mask, square=True, cmap='crest')
        plt.show()
    except Exception as e:
        print(str(e))
    