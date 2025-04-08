import pandas as pd
import numpy as np

import matplotlib.pylab as plt

from utility import print_table, import_csv     #, backup_csv


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
filename = "usedCars.csv"
headers = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors", "body-style",
           "drive-wheels", "engine-location", "wheel-base", "length", "width", "height", "curb-weight", "engine-type",
           "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke", "compression-ratio", "horsepower",
           "peak-rpm", "city-mpg", "highway-mpg", "price"]

# Using custom function for importing the CSV
df = import_csv(filename, url, headers) # To create a backup: back_up(filename, url)


######################### DEALING WITH MISSING DATAS #########################


# Replace '?' with nan
df = df.replace('?', np.nan)  # Removed inplace = True

# Understanding missing datas by looking at them
"""
# Creating a dataframe of missing values: returns a data frame with 0/1, 1 where value is nan
missing_data = df.isnull()
print_table(missing_data.head(1))

# Counting number of missing values in each column
for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts().to_dict())
    print("")
"""

# Replacing missing values with mean
for column in ["normalized-losses", "stroke", "bore", "horsepower", "peak-rpm"]:
    mean = df[column].astype("float").mean(axis=0)
    df[column] = df[column].replace(np.nan, mean)  # Removed inplace=True

# Replacing the missing 'num-of-doors' values by the most frequent
frequent = str(df["num-of-doors"].value_counts().idxmax())
df["num-of-doors"] = df["num-of-doors"].replace(np.nan, frequent)

# Drop rows where "price" is missing
df = df.dropna(subset=["price"], axis=0)
df = df.reset_index(drop=True)      # Reset index after dropping rows (optional)

# TEST
print_table(df)


######################### CORRECTING DATA FORMATS #########################


# Double brackets are only necessary when working with multiple columns at once: df["price"] = df["price"].astype("float") would also work
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")

# TEST
print_table(df.dtypes)


######################### DATA STANDARDIZATION #########################


# Unit conversion: L/100km = 235 / mpg
df["city-L/100km"] = 235/df["city-mpg"]                         # Adds columns named "city-L/100km"

df["highway-mpg"] = 235/df["highway-mpg"]                       # Replaces column value
df = df.rename(columns = {"highway-mpg": "highway-L/100km"})    #  Replaces column name as well as values


######################### DATA NORMALIZATION #########################


# replace (original value) by (original value)/(maximum value)
df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()
df['height'] = df['height']/df['height'].max()

# TEST
df[["length","width","height"]].head()


######################### BINNING #########################


df["horsepower"] = df["horsepower"].astype("int", copy=True)

bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
group_names = ['Low', 'Medium', 'High']
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True)

# TEST
print_table(df[['horsepower','horsepower-binned']].head(20))
print_table(df["horsepower-binned"].value_counts())


# Draw histogram of attribute "horsepower" with bins = 3
plt.hist(df["horsepower"], bins=3)

# Set x/y labels and plot title
plt.xlabel("horsepower")
plt.ylabel("count")
plt.title("horsepower bins")

# Show the plot in a separate window
plt.show()


######################### INDICATOR VARIABLE #########################


dummy_variable_1 = pd.get_dummies(df["fuel-type"])
# print_table(dummy_variable_1.head(10))

dummy_variable_1 = dummy_variable_1.rename(columns={'gas':'fuel-type-gas', 'diesel':'fuel-type-diesel'})
# print_table(dummy_variable_1.head(10))

# merge data frame "df" and "dummy_variable_1"
df = pd.concat([df, dummy_variable_1], axis=1)

# drop original column "fuel-type" from "df"
df = df.drop("fuel-type", axis = 1)


# get indicator variables of aspiration and assign it to data frame "dummy_variable_2"
dummy_variable_2 = pd.get_dummies(df['aspiration'])

# change column names for clarity
dummy_variable_2 = dummy_variable_2.rename(columns={'std':'aspiration-std', 'turbo': 'aspiration-turbo'})

# show first 5 instances of data frame "dummy_variable_1"
dummy_variable_2.head()

# merge the new dataframe to the original dataframe
df = pd.concat([df, dummy_variable_2], axis=1)

# drop original column "aspiration" from "df"
df = df.drop('aspiration', axis = 1)


df.to_csv('clean_df.csv')


########################################################################################################################