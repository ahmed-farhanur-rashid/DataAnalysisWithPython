import numpy as np
import pandas as pd
import os

# noinspection PyUnresolvedReferences
from pandasgui import show

from utility import print_table, import_csv


pd.set_option('display.max_columns', None)


url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_base.csv"

directory = r"C:\Users\ASUS\PycharmProjects\DataAnalysisWithPython\Module 1"
filename = "laptop_pricing_dataset_base.csv"

filePath = os.path.join(directory, filename)


headers = ["Manufacturer", "Category", "Screen", "GPU", "OS", "CPU_core","Screen_Size_inch",
           "CPU_frequency", "RAM_GB", "Storage_GB_SSD", "Weight_kg", "Price"]

# Load the dataset to a pandas dataframe named 'df'
# Add headers to the dataframe. Alt:         df.columns = headers
# Replace '?' with 'NaN'
if os.path.exists(filePath):
    df = pd.read_csv(filePath)
else:
    df = pd.read_csv(url, names=headers)
    df = df.replace('?', np.nan)
    df.to_csv(filePath, index=False)

# Load first 10 rows
print_table(df.head(10))
print("")

# Print the data types of the dataframe columns
print_table(df.dtypes)
print("")

# Print the statistical description of the dataset, including that of 'object' data types.
print_table(df.describe(include="all"))
print("")

# Print the summary information of the dataset.
print_table(df.info())
print("")

# Show Data Frame in PandasGUI
# show(df)