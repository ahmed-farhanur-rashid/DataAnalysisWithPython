import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

data = {
    'City': ['A', 'A', 'B', 'B', 'C', 'C', 'A', 'B', 'C'],
    'Day':  ['Mon', 'Tue', 'Mon', 'Tue', 'Mon', 'Tue', 'Wed', 'Wed', 'Wed'],
    'Temp': [23, 25, 22, 24, 20, 21, 26, 27, 22]
}

df = pd.DataFrame(data)

pivot = df.pivot_table(values='Temp', index='Day', columns='City')

plt.pcolor(pivot, cmap='viridis')
plt.colorbar()  # Optional: shows the scale of color → value
plt.xticks(ticks=[0.5, 1.5, 2.5], labels=pivot.columns)  # Align x-labels
plt.yticks(ticks=[0.5, 1.5, 2.5], labels=pivot.index)    # Align y-labels
plt.title("Temperature by Day and City")
plt.show()



