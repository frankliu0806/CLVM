import pandas as pd

#load firm data and define Y
firm  = pd.read_csv("processed_data.csv")
Y_field = ['Total revenue - 2017']
all_field = list(firm.columns)
all_field.remove('Group') #remove categorical variables here 

X_field = list(all_field)
X_field.remove('Total revenue - 2017')

#sorted_firm = pd.concat([firm[X_field], firm[Y_field]], axis = 1)

corr_table =firm.corr(method='pearson').abs()
#print(corr_table)
#print("\n")


corr_cutoff = 0.3

print("corr_cutoff = %f\n" %corr_cutoff)

for x in X_field:
    for y in X_field:
        if corr_table[x][y]>corr_cutoff and X_field.index(x)<X_field.index(y):
            print("Correlation of %s and %s is %f" %(x, y, corr_table[x][y]))
            print("Correlation of %s and Y is %f" %(x, corr_table[x][Y_field]))
            print("Correlation of %s and Y is %f" %(y, corr_table[y][Y_field]))
            print("\n")
            
print("SFA done") 

#sorted pairwise correlation
"""
stack = corr_table.unstack()
sorted_stack = stack.sort_values(kind="quicksort", ascending = False)
print sorted_stack.head(20)
"""