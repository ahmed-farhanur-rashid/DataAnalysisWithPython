import pandas as pd
import numpy as np
import seaborn as sns
import dtale

import matplotlib.pylab as plt
from pandasgui import show
from utility import print_table, import_csv, show_table_with_matplotlib


url = ("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
       "IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv")
filename = "automobileEDA.csv"

df = import_csv(filename, url)

""""

print_table(df.dtypes)

numeric_dataFrame = df.select_dtypes(include=["int64", "float64"])
print_table(numeric_dataFrame.corr())

"""

################ What is the data type of the column "peak-rpm"? ################
print(df["peak-rpm"].dtypes)

############ Find the correlation between the following columns: bore, stroke, compression-ratio, and horsepower. ################
print_table(df[["bore", "stroke", "compression-ratio", "horsepower"]].corr())

"""

# Positive Strong Correlation
sns.regplot(x = "engine-size", y = "price", data=df)
plt.ylim(0,)
plt.show()
print_table(df[["engine-size", "price"]].corr())

# Negative Strong Correlation
sns.regplot(x = "highway-mpg", y = "price", data = df)
plt.ylim(0,)
plt.show()
print_table(df[["highway-mpg", "price"]].corr())

# Weak Correlation
sns.regplot(x = "peak-rpm", y = "price", data = df)
plt.ylim(0,)
plt.show()
print_table(df[["peak-rpm", "price"]].corr())

"""

################ Find the correlation between x="stroke" and y="price". ################

# There is a weak correlation between the variable 'stroke' and 'price.' as such regression will not work well.
# We can see this using "regplot" to demonstrate this.

sns.regplot(x = "stroke", y = "price", data = df)
plt.ylim(0,)
plt.show()
print_table(df[["stroke", "price"]].corr())

#"""

# Categorical variable plotting
sns.boxplot(x = "body-style", y = "price", data = df)
plt.title("Boxplot of Price by Body Style")
plt.show()

sns.boxplot(x="engine-location", y="price", data=df)
plt.title("Boxplot of Price by Engine Location")
plt.show()









#"""




