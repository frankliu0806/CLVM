import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

#load firm data and define Y
firm  = pd.read_csv("processed_data.csv")
Y = np.asarray(firm['Total revenue - 2017'].values)


#Box-cox transformation for Y
lmax = stats.boxcox_normmax(Y, brack = (-3, 0))
print("lmax = %f" %lmax)
Yt = stats.boxcox(Y, lmax)


#Comparison of Y and Yt 
fig = plt.figure()
ax1 = fig.add_subplot(211)
prob = stats.probplot(Y, dist=stats.norm, plot=ax1)
ax1.set_xlabel('')
ax1.set_title('Probplot against normal distribution')

ax2 = fig.add_subplot(212)
prob = stats.probplot(Yt, dist=stats.norm, plot=ax2)
ax2.set_title('Probplot after Box-Cox transformation')

#plt.show()

#Check if the result is local optimal by plotting log likelihood - lambda plot
lmbdas = np.linspace(-10, 10)

llf = np.zeros(lmbdas.shape, dtype=float)
for ii, lmbda in enumerate(lmbdas):
    llf[ii] = stats.boxcox_llf(lmbda, Y)

Y_most_normal, lmbda_optimal = stats.boxcox(Y)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(lmbdas, llf, 'b.-')
ax.axhline(stats.boxcox_llf(lmbda_optimal, Y), color='r')
ax.set_xlabel('lmbda parameter')
ax.set_ylabel('Box-Cox log-likelihood')

#plt.show()

#Transform X 
X_fields = ['Cost', 'Firm age', 'Firm size', 'Firm # floors', 'Firm avg # staff', 'Firm # manager']

for x in X_fields:
    
    #transformation of X
    f1 = np.log(firm[x]+abs(firm[x].min())+1) 
    f2 = np.sign(firm[x])*np.sqrt(abs(firm[x]))
    
    tmp = firm[x].clip(lower = 0.00001)
    f3 = 1 / tmp
    
    f4 = np.square(firm[x])
    f5 = np.exp(firm[x])
    
    #correlation coefficient against original Y
    r0 = abs(np.corrcoef(firm[x],Y)[1,0])
    r1 = abs(np.corrcoef(f1, Y)[1,0])
    r2 = abs(np.corrcoef(f2, Y)[1,0])
    r3 = abs(np.corrcoef(f3, Y)[1,0])
    r4 = abs(np.corrcoef(f4, Y)[1,0])
    r5 = abs(np.corrcoef(f5, Y)[1,0])
    
    #correlation coefficient against transformed Y
    s0 = abs(np.corrcoef(firm[x],Yt)[1,0])
    s1 = abs(np.corrcoef(f1, Yt)[1,0])
    s2 = abs(np.corrcoef(f2, Yt)[1,0])
    s3 = abs(np.corrcoef(f3, Yt)[1,0])
    s4 = abs(np.corrcoef(f4, Yt)[1,0])
    s5 = abs(np.corrcoef(f5, Yt)[1,0])
    
    
    #create correlation coefficient table
    trans_type =pd.DataFrame(['X', 'log', 'sqrt', '1/X','^2','exp'])
    corr_Y = pd.DataFrame([r0, r1, r2, r3, r4, r5])
    corr_Yt = pd.DataFrame([s0, s1, s2, s3, s4, s5])
    corr_table = pd.concat([trans_type, corr_Y, corr_Yt], axis = 1)
    corr_table.columns = ['','Y','Yt']
    print("\n")
    print(x)
    print(corr_table)
    
    




    
    
    
    