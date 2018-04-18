#!/usr/bin/env python
from kmodes.kprototypes import KPrototypes
import pandas as pd
from scipy import stats


# stocks with their market caps, sectors and countries
DataX = pd.read_csv('input.csv')

X = DataX.iloc[:,[1,2,3]]
X_values = X.values

n_clusters = 4

kproto = KPrototypes(n_clusters=n_clusters, init='Cao', verbose=8)
clusters = kproto.fit_predict(X_values, categorical=[1, 2]) #TC: define categorical variables here

# Print cluster centroids of the trained model.
print("\nCluster centroid")
df = pd.DataFrame(kproto.cluster_centroids_[0])
df1 = pd.DataFrame(kproto.cluster_centroids_[1])
centroid = pd.concat([df, df1], axis = 1)
centroid.columns = ['market_cap','sector','country']
print(centroid)


# Print training statistics
print("\nCost")
print(kproto.cost_)
print("\nNumber of iterations")
print(kproto.n_iter_)


print("\nClustering result")
DataX['cluster']=clusters
print(DataX)

#ttest
print("\nT-test")
groupby_cluster = DataX.groupby('cluster')

market_cap = {}

for i in range(0, n_clusters):
    market_cap[i] = groupby_cluster.get_group(i)

for j in range(0, n_clusters):
    for k in range(j+1, n_clusters):
        print '\ncluster %i vs cluster %i' %(j,k)
        if (len(market_cap[j])>1 and len(market_cap[k])>1):
            print(stats.ttest_ind(market_cap[j]['market_cap'], market_cap[k]['market_cap']))
        else:
            print("At least one cluster has length == 1")


