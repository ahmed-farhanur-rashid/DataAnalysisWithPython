import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utility import import_csv, backup_csv, print_table


url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod1.csv"
filename = "laptops.csv"
headers = ["Manufacturer", "Category", "Screen", "GPU", "OS", "CPU_core", "Screen_Size_cm",
           "CPU_frequency", "RAM_GB", "Storage_GB_SSD", "Weight_kg", "Price"]

import_csv(filename, url)
backup_csv(filename, url)

df = pd.read_csv(filename, names = headers, header = 0)
print_table(df)

# To round up a certain column upto 2 decimals & to set nan
df[['Screen_Size_cm']] = np.round(df[['Screen_Size_cm']],2)
df = df.replace('?', np.nan)


# Task - 1: Evaluate the dataset for missing data


missing_data = df.isnull()
for column in missing_data.columns.values.tolist():
    print(column)
    print_table(missing_data[column].value_counts())


# Task - 2: Handling missing data (Weight) (Screen Size)


column = "Weight_kg"
mean = df[column].astype("float").mean()
df[column] = df[column].replace(np.nan, mean)

column = "Screen_Size_cm"
frequent = str(df[column].value_counts().idxmax())
df[column] = df[column].replace(np.nan, frequent)


# Task - 3: Fixing data types


df["Weight_kg"] = df["Weight_kg"].astype("float")
df["Screen_Size_cm"] = df["Screen_Size_cm"].astype("float")


# Task 4: Data Standardization & Normalization


df["Weight_kg"] = df["Weight_kg"] * 2.205
df["Screen_Size_cm"] = df["Screen_Size_cm"] / 2.54

df = df.rename(columns={"Weight_kg": "Weight_pounds"})
df = df.rename(columns={"Screen_Size_cm": "Screen_Size_inch"})

df["CPU_frequency"] = df["CPU_frequency"] / df["CPU_frequency"].max()


# Task 5: Binning


df["Price"] = df["Price"].astype("float")
bins = np.linspace(df["Price"].min(), df["Price"].max(), 4)
group_names = ['Low', 'Medium', 'High']
df["Price-binned"] = pd.cut(df["Price"], bins, labels=group_names, include_lowest=True)

plt.bar(group_names, df["Price-binned"].value_counts())
plt.xlabel("Price")
plt.ylabel("count")
plt.title("Price bins")
plt.show()


# Task 6: Indicator variables


dummy_variable = pd.get_dummies(df["Screen"])
dummy_variable = dummy_variable.rename(columns={"IPS Panel": "Screen-IPS_panel", "Full HD": "Screen-Full_HD"})
df = pd.concat([df, dummy_variable], axis=1)
df = df.drop("Screen", axis = 1)

print_table(df)

df.to_csv("Laptops (Cleaned).csv")
