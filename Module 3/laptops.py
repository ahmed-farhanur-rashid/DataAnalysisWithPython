import os
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from utility import print_table
from scipy import stats

url = ("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
       "IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod2.csv")
file_name="laptops.csv"

if not os.path.exists(file_name):
    download = pd.read_csv(url)
    download.drop(columns=["Unnamed: 0", "Unnamed: 0.1"], inplace=True)
    download.to_csv(file_name, index=False)

# Read File
df = pd.read_csv(file_name, header = 0)



#           Task - 1
# Generate regression plots for each of the parameters
#       "CPU_frequency"
#       "Screen_Size_inch"
#       "Weight_pounds"
#               against
#                   "Price".
#       Print the value of correlation of each feature with "Price".

sns.regplot(x = "CPU_frequency", y = "Price", data = df)
plt.ylim(0,)
plt.show()

sns.regplot(x = "Screen_Size_inch", y = "Price", data = df)
plt.ylim(0,)
plt.show()

sns.regplot(x = "Weight_pounds", y = "Price", data = df)
plt.ylim(0,)
plt.show()

for parameter in ["CPU_frequency", "Screen_Size_inch", "Weight_pounds"]:
    print(f"Correlation of Price and {parameter} is ", df[[parameter,"Price"]].corr())

# Generate Box plots for the different feature that hold categorical values.
# These features would be "Category", "GPU", "OS", "CPU_core", "RAM_GB", "Storage_GB_SSD"

sns.boxplot(x = "Category", y = "Price", data = df)
plt.title("CPU Frequency vs Laptop Price")
plt.show()

sns.boxplot(x = "GPU", y = "Price", data = df)
plt.title("GPU vs Laptop Price")
plt.show()

sns.boxplot(x = "OS", y = "Price", data = df)
plt.title("OS vs Laptop Price")
plt.show()

sns.boxplot(x = "CPU_core", y = "Price", data = df)
plt.title("CPU_core vs Laptop Price")
plt.show()

sns.boxplot(x = "RAM_GB", y = "Price", data = df)
plt.title("RAM_GB vs Laptop Price")
plt.show()

sns.boxplot(x = "Storage_GB_SSD", y = "Price", data = df)
plt.title("Storage_GB_SSD vs Laptop Price")
plt.show()



#           Task - 2
# Generate the statistical description of all the features being used in the data set.
# Include "object" data types as well.

print_table(df.describe())
print_table(df.describe(include = "object"))



#           Task - 3
# Group the parameters "GPU", "CPU_core" and "Price" to make a pivot table
# Visualize this connection using the pcolor plot.

temp = df[['GPU','CPU_core','Price']]
temp = temp.groupby(['GPU','CPU_core'],as_index=False).mean()

grouped_pivot = temp.pivot(index = 'GPU', columns = 'CPU_core', values = 'Price')
grouped_pivot = grouped_pivot.fillna(0)

# Creating Heatmap
figure, axis = plt.subplots()
image = axis.pcolor(grouped_pivot, cmap='RdBu')

# Set general labels
axis.set_xlabel("CPU Core Count")
axis.set_ylabel("GPU Type")
axis.set_title("Average Laptop Price by GPU and CPU Cores")

# Label names
row_labels = grouped_pivot.columns
col_labels = grouped_pivot.index

# Move ticks and labels to the center
axis.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
axis.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

# Insert labels
axis.set_xticklabels(row_labels, minor=False)
axis.set_yticklabels(col_labels, minor=False)

# Rotate x labels for readability
plt.xticks(rotation=90)

# Add colorbar
figure.colorbar(image)
plt.show()



#           Task - 4
# Use the scipy.stats.pearsonr() function to evaluate the Pearson Coefficient and the p-values for each parameter tested above.
# This will help you determine the parameters most likely to have a strong effect on the price of the laptops.

for parameter in ['RAM_GB','CPU_frequency','Storage_GB_SSD','Screen_Size_inch','Weight_pounds','CPU_core','OS','GPU','Category']:
    pearson_coefficient, p_value = stats.pearsonr(df[parameter], df['Price'])
    print("The Pearson Correlation Coefficient is ", pearson_coefficient,
          " with a P-value of P = ", p_value)



