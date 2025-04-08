import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pylab as plt
from utility import print_table, import_csv
from scipy import stats


url = ("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
       "IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv")
filename = "automobileEDA.csv"

df = import_csv(filename, url)

""""

# Start


print_table(df.dtypes)

numeric_dataFrame = df.select_dtypes(include=["int64", "float64"])
print_table(numeric_dataFrame.corr())

"""

################ What is the data type of the column "peak-rpm"? ################
print(df["peak-rpm"].dtypes)

############ Find the correlation between the following columns: bore, stroke, compression-ratio, and horsepower. ################
print_table(df[["bore", "stroke", "compression-ratio", "horsepower"]].corr())

"""

# Continuous Numerical Variables


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

"""

# Categorical variable plotting


sns.boxplot(x = "body-style", y = "price", data = df)
plt.title("Boxplot of Price by Body Style")
plt.show()

sns.boxplot(x="engine-location", y="price", data=df)
plt.title("Boxplot of Price by Engine Location")
plt.show()

sns.boxplot(x="drive-wheels", y="price", data=df)
plt.title("Boxplot of Price by Drive Wheels")
plt.show()

"""

"""

# Value Counts


print_table(df.describe(include = "object"))

# drive-wheels as variable
drive_wheels_counts = df['drive-wheels'].value_counts().to_frame().reset_index()
drive_wheels_counts.reset_index(inplace=True)
drive_wheels_counts.index.name = 'drive-wheels'
print_table(drive_wheels_counts)

# engine-location as variable
engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
print_table(engine_loc_counts)

"""

# Basics of Grouping


print(df['drive-wheels'].unique())

df_group_1 = df[['drive-wheels','body-style','price']]

#      df_grouped = df_group_1.groupby(['drive-wheels'], as_index=False).agg({'price': 'mean'})
#      df_grouped = df_group_1.groupby(['drive-wheels', 'body-style'], as_index=False).agg({'price': 'mean'})

df_grouped = df_group_1.groupby(['drive-wheels', 'body-style'], as_index=False).mean()

print(df_grouped)

grouped_pivot = df_grouped.pivot(index='drive-wheels', columns='body-style', values='price')
grouped_pivot = grouped_pivot.fillna(0)


################ Use the "groupby" function to find the average "price" of each car based on "body-style". ################
temp = df[["body-style", "price"]]
grouped_by_body_style = temp.groupby(['body-style'], as_index=False).mean()
print_table(grouped_by_body_style)


# Create the heatmap
fig, ax = plt.subplots(figsize=(12, 6))  # Wider figure to fit labels
im = ax.pcolor(grouped_pivot, cmap='RdBu')

# Set label names
col_labels = grouped_pivot.columns  # X-axis (body-style)
row_labels = grouped_pivot.index    # Y-axis (drive-wheels)

# Set ticks
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

# Insert labels
ax.set_xticklabels(col_labels, minor=False)
ax.set_yticklabels(row_labels, minor=False)

# Rotate x labels for readability
plt.xticks(rotation=90)

# Add colorbar
fig.colorbar(im)

# Fix layout to prevent clipping
plt.tight_layout()
plt.show()


######## Correlation and Causation ########

print_table(df.select_dtypes(include=['number']).corr())

pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)

pearson_coef, p_value = stats.pearsonr(df['length'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)

pearson_coef, p_value = stats.pearsonr(df['width'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value )

pearson_coef, p_value = stats.pearsonr(df['curb-weight'], df['price'])
print( "The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)

pearson_coef, p_value = stats.pearsonr(df['engine-size'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

pearson_coef, p_value = stats.pearsonr(df['bore'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =  ", p_value )

pearson_coef, p_value = stats.pearsonr(df['city-mpg'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)

pearson_coef, p_value = stats.pearsonr(df['highway-mpg'], df['price'])
print( "The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value )