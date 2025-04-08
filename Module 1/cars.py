import os

import pandas
import numpy

#pandas.set_option('display.max_columns', None)

#Functions
def get_dataset(target_filepath, target_url):

    if not os.path.exists(filepath):
        print("File not found. Downloading...\n")
        target_df = pandas.read_csv(target_url, header=None)
        target_df.to_csv(target_filepath, index=False)
        print(f"File saved to {target_filepath}")
    else:
        print("File already exists. Loading from disk...\n")
        target_df = pandas.read_csv(target_filepath)

    return target_df # df = pandas.read_csv(target_filepath)



# Define the file path and URL
directory = r"C:\Users\ASUS\PycharmProjects\DataAnalysisWithPython\Module 1"
filename = "cars.csv"
filepath = os.path.join(directory, filename)
url = r"https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

# Create headers list
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

# Getting the dataset as DataFrame
df = get_dataset(filepath, url)
df.columns = headers

# Creating Modified DataFrame by replacing '?' with NaN
data_frame = df.replace('?',numpy.nan)

# Drooping entire row whenever that row has a Price value of NaN
df = data_frame.dropna(subset=["price"], axis=0)

# To save a copy in Home directory
df.to_csv("automobile.csv", index=False)

print(df.head(20))

print(df.columns)
print(df.dtypes)
print(df.describe())
print(df.describe(include = "all"))
print(df[['length', 'compression-ratio']].describe())
print(df.info())

#print("The first 10 rows of the dataframe")
#print(df.head(10))
#print("")
#print("The last 10 rows of the dataframe")
#print(df.tail(10))




