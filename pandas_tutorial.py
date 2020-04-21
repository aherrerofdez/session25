import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# OBJECT CREATION
# Creating a Series by passing a list of values, letting pandas create a default integer index
s = pd.Series([1, 3, 5, np.nan, 6, 8])
print(s)
# Creating a DataFrame by passing a NumPy array, with a datetime index and labeled columns
dates = pd.date_range('20130101', periods=6)
print(dates)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
print(df)
# Creating a DataFrame by passing a dict of objects that can be converted to series-like
df2 = pd.DataFrame({'A': 1.,
                    'B': pd.Timestamp('20130102'),
                    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                    'D': np.array([3] * 4, dtype='int32'),
                    'E': pd.Categorical(["test", "train", "test", "train"]),
                    'F': 'foo'})
print(df2)
# The columns of the resulting DataFrame have different dtypes
print(df2.dtypes)

# VIEWING DATA
# How to view the top and bottom rows of the frame
print(df.head())
print(df.tail(3))
# Display the index and columns
print(df.index)
print(df.columns)
# Get a NumPy representation of the underlying data
print(df.to_numpy())
print(df2.to_numpy())
# Show a quick statistic summary of the data
print(df.describe())
# Transposing the data
print(df.T)
# Sorting by an index
print(df.sort_index(axis=1, ascending=False))
# Sorting by values
print(df.sort_values(by='B'))

# SELECTION
# Selecting a single column
print(df['A'])
# Selecting via [], which slices the rows
print(df[0:3])
print(df['20130102':'20130104'])
# Getting a cross section using a label
print(df.loc[dates[0]])
# Selecting on a multi-axis by label
print(df.loc[:, ['A', 'B']])
# Showing label slicing, both endpoints are included
print(df.loc['20130102':'20130104', ['A', 'B']])
# Reduction in the dimensions of the returned object
print(df.loc['20130102', ['A', 'B']])
# For getting a scalar value
print(df.loc[dates[0], 'A'])
# For getting fast access to a scalar (equivalent to prior method)
print(df.at[dates[0], 'A'])
# Selecting via the position of the passed integers
print(df.iloc[3])
# Selecting by integer slices, acting similar to numpy / python
print(df.iloc[3:5, 0:2])
# Selecting by lists of integer position locations, similar to the numpy / python style
print(df.iloc[[1, 2, 4], [0, 2]])
# For slicing rows explicitly
print(df.iloc[1:3, :])
# For slicing columns explicitly
print(df.iloc[:, 1:3])
# For getting a value explicitly
print(df.iloc[1, 1])
# For getting a fast access to a scalar (equivalent to prior method)
print(df.iat[1, 1])
# Using a single column's values to select data
print(df[df['A'] > 0])
# Selecting values from a DataFrame where a boolean condition is met
print(df[df > 0])
# Using the isin() method for filtering
df2 = df.copy()
df2['E'] = ['one', 'one', 'two', 'three', 'four', 'three']
print(df2)
print(df2[df2['E'].isin(['two', 'four'])])
# Setting a new column automatically aligns the data by the indexes
s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('20130102', periods=6))
print(s1)
df['F'] = s1
# Setting values by label
df.at[dates[0], 'A'] = 0
# Setting values by position
df.iat[0, 1] = 0
# Setting by assigning with a NumPy array
df.loc[:, 'D'] = np.array([5] * len(df))
print(df)
# A 'where' operation with setting
df2 = df.copy()
df2[df2 > 0] = -df2
print(df2)

# MISSING DATA
# Reindexing allows to change / add / delete the index on a specified axis. This returns a copy of the data
df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])
df1.loc[dates[0]:dates[1], 'E'] = 1
print(df1)
# To drop any rows that have missing data
print(df1.dropna(how='any'))
# Filling missing data
print(df1.fillna(value=5))
# To get the boolean mask where values are NaN
print(pd.isna(df1))

# OPERATIONS
# Performing a prescriptive statistic
print(df.mean())
# Same operation on the other axis
print(df.mean(1))
# Operating with objects that have different dimensionality and need alignment
# Pandas automatically broadcasts along the specified dimension
s = pd.Series([1, 3, 5, np.nan, 6, 8], index=dates).shift(2)
print(s)
print(df.sub(s, axis='index'))
# Applying functions to the data
print(df.apply(np.cumsum))
print(df.apply(lambda x: x.max() - x.min()))
# Histogramming and Discretization
s = pd.Series(np.random.randint(0, 7, size=10))
print(s)
print(s.value_counts())
# String Processing Methods
s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
print(s.str.lower())

# MERGE
# Concatenating pandas objects together with concat()
df = pd.DataFrame(np.random.randn(10, 4))
print(df)
# Break it into pieces and concatenate
pieces = [df[:3], df[3:7], df[7:]]
print(pd.concat(pieces))
# SQL style merging
left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})
print(left)
print(right)
print(pd.merge(left, right, on='key'))
left = pd.DataFrame({'key': ['foo', 'bar'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'bar'], 'rval': [4, 5]})
print(left)
print(right)
print(pd.merge(left, right, on='key'))

# GROUPING
# Splitting the data into groups based on some criteria
# Applying a function to each group independently
# Combining the results into a data structure
df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
                   'B': ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
                   'C': np.random.randn(8),
                   'D': np.random.randn(8)})
print(df)
print(df.groupby('A').sum())
print(df.groupby(['A', 'B']).sum())

# RESHAPING
# Stack method compresses a level in the DataFrame's columns
tuples = list(zip(*[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                    ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']]))
index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=['A', 'B'])
df2 = df[:4]
print(df2)
stacked = df2.stack()
print(stacked)
print(stacked.unstack())
print(stacked.unstack(1))
print(stacked.unstack(0))
# Pivot Tables
df = pd.DataFrame({'A': ['one', 'one', 'two', 'three'] * 3,
                   'B': ['A', 'B', 'C'] * 4,
                   'C': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
                   'D': np.random.randn(12),
                   'E': np.random.randn(12)})
print(df)
print(pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C']))

# TIME SERIES
# Performing resampling operations during frequency conversion
rng = pd.date_range('1/1/2012', periods=100, freq='S')
ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
print(ts.resample('5Min').sum())
# Time zone representation
rng = pd.date_range('3/6/2012 00:00', periods=5, freq='D')
ts = pd.Series(np.random.randn(len(rng)), rng)
print(ts)
ts_utc = ts.tz_localize('UTC')
print(ts_utc)
# Converting to another time zone
print(ts_utc.tz_convert('US/Eastern'))
# Converting between time span representation
rng = pd.date_range('1/1/2012', periods=5, freq='M')
ts = pd.Series(np.random.randn(len(rng)), index=rng)
print(ts)
ps = ts.to_period()
print(ps)
print(ps.to_timestamp())
# Converting between period and timestamp enables some convenient arithmetic functions to be used
prng = pd.period_range('1990Q1', '2000Q4', freq='Q-NOV')
ts = pd.Series(np.random.randn(len(prng)), prng)
ts.index = (prng.asfreq('M', 'e') + 1).asfreq('H', 's') + 9
print(ts.head())

# CATEGORICALS
# Including categorical data in DataFrame
df = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6],
                   "raw_grade": ['a', 'b', 'b', 'a', 'a', 'e']})
# Converting the raw grades to a categorical data type
df["grade"] = df["raw_grade"].astype("category")
print(df["grade"])
# Renaming the categories to more meaningful names
df["grade"].cat.categories = ["very good", "good", "very bad"]
# Reorder the categories and add the missing categories
df["grade"] = df["grade"].cat.set_categories(["very bad", "bad", "medium", "good", "very good"])
print(df["grade"])
# Sorting is per order in the categories, not lexical order
print(df.sort_values(by="grade"))
# Grouping by a categorical column also shows empty categories
print(df.groupby("grade").size())

# PLOTTING
plt.close('all')
ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()
ts.plot()
#plt.show()
df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index, columns=['A', 'B', 'C', 'D'])
df = df.cumsum()
plt.figure() # Blank Figure
df.plot()
plt.legend(loc='best')
plt.show()


# GETTING DATA IN / OUT
# Writing to a csv file
df.to_csv('foo.csv')
# Reading from a csv file
print(pd.read_csv('foo.csv'))
# Writing to a HDF5 Store
df.to_hdf('foo.h5', 'df')
# Reading from a HDF5 Store
print(pd.read_hdf('foo.h5', 'df'))
# Writing to an excel file
df.to_excel('foo.xlsx', sheet_name='Sheet1')
# Reading from an excel file
print(pd.read_excel('foo.xlsx', 'Sheet1', index_col=None, na_values=['NA']))
