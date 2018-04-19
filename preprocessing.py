import pandas as pd
from scipy.stats.mstats import winsorize

#initialize firm
firm = pd.read_csv("input.csv")
firm['Firm ID'] = firm['Firm ID'].astype('category',copy = False)
firm['Group'] = firm['Group'].astype('category',copy = False)


print("Original")
print(firm.tail())

#null handling
handled_columns = ['Firm age', 'Firm size', 'Firm # floors']
firm[handled_columns] = firm[handled_columns].fillna(firm[handled_columns].median())
print("\nNull handling - Replace NaN with median")
print(firm.tail())

#winsorization
winsorized_columns = ['Cost', 'Firm age']
firm.loc[:, winsorized_columns] =  winsorize(firm.loc[:,winsorized_columns].values, limits=[0.05, 0.05])
print("\nWinsorized columns %s" %winsorized_columns)
print(firm.head())

#standardization
standardized_columns = ['Firm age', 'Firm size']
firm_std = firm
firm_std[standardized_columns] = (firm[standardized_columns] - firm[standardized_columns].mean()) / firm[standardized_columns].std()
print("\nStandardized")
print(firm_std.head())
