import pandas as pd
import os

from matplotlib import pyplot as plt
from tabulate import tabulate


def print_table(tdf):
    # Check if tdf is a Pandas DataFrame
    if isinstance(tdf, pd.DataFrame):
        print(tabulate(tdf, headers='keys', tablefmt='psql'))
    elif isinstance(tdf, pd.Series):
        # If it's a Series (like df.dtypes), convert it to a DataFrame
        print(tabulate(tdf.reset_index(), headers=['Column', 'Data Type'], tablefmt='psql'))
    else:
        # If it's not a DataFrame or Series (like df.info()), just print it as is
        print(tdf)
    print("")  # Add spacing for better readability


def import_csv(filename, url=None, headers=None):
    # Check if the file exists in the current working directory
    if os.path.exists(filename):
        df = pd.read_csv(filename)
    elif url:                               # If the file doesn't exist, and a URL is provided
        df = pd.read_csv(url, names=headers if headers else None)
        df.to_csv(filename, index=False)    # Save the CSV file locally if it was fetched from the URL
    else:
        raise FileNotFoundError(f"{filename} not found, and no URL provided to download it.")

    return df

def backup_csv(filename, url):
    # Check if the file exists in the current working directory
    if os.path.exists(filename):
        df = pd.read_csv(url, header = None)
        backup_filename = f"backup_{filename}"
        df.to_csv(backup_filename, index=False)

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

def show_table_with_matplotlib(tdf):
    if isinstance(tdf, pd.DataFrame):
        # Get dimensions to size the figure dynamically
        n_rows, n_cols = tdf.shape
        fig, ax = plt.subplots(figsize=(n_cols * 2.5, n_rows * 0.6 + 2))  # Dynamic size

        ax.axis('off')

        # Build the table
        table = ax.table(
            cellText=tdf.round(2).values,
            rowLabels=tdf.index,
            colLabels=tdf.columns,
            cellLoc='center',
            loc='center'
        )

        # Optional tweaks
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.2)

        # Title (optional)
        plt.title("Correlation Table", fontsize=14, pad=20)

        # Let the table control the size
        plt.tight_layout()
        plt.savefig("correlation_table.png", bbox_inches='tight')  # Optional: save to file
        plt.show()

