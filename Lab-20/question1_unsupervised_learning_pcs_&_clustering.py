import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.datasets import get_rdataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ISLP import load_data

from sklearn.cluster import \
(KMeans ,
AgglomerativeClustering)
from scipy.cluster.hierarchy import \
(dendrogram ,
cut_tree)
from ISLP.cluster import compute_linkage
USArrests = get_rdataset('USArrests').data
USArrests.head()
print(USArrests.columns)

print(USArrests.mean())
print(USArrests.var())

scaler = StandardScaler(with_std=True ,with_mean=True)
USArrests_scaled = scaler.fit_transform(USArrests)
pcaUS = PCA()
pcaUS.fit(USArrests_scaled)
print(pcaUS.mean_)

scores = pcaUS.transform(USArrests_scaled)
print(pcaUS.components_)

i, j = 0, 1 # which components
fig , ax = plt.subplots (1, 1, figsize =(8, 8))
ax.scatter(scores [:,0], scores [: ,1])
ax.set_xlabel('PC%d' % (i+1))
ax.set_ylabel('PC%d' % (j+1))
for k in range(pcaUS.components_.shape [1]):
    ax.arrow(0, 0, pcaUS.components_[i, k], pcaUS.components_[j, k])
    ax.text(pcaUS.components_[i, k],
            pcaUS.components_[j, k],
            USArrests.columns[k])

scale_arrow = s_ = 2
scores [:,1] *= -1
pcaUS.components_ [1] *= -1 # flip the y-axis
fig , ax = plt.subplots (1, 1, figsize =(8, 8))
ax.scatter(scores [:,0], scores [: ,1])
ax.set_xlabel('PC%d' % (i+1))
ax.set_ylabel('PC%d' % (j+1))
for k in range(pcaUS.components_.shape [1]):
    ax.arrow(0, 0, s_*pcaUS.components_[i,k], s_*pcaUS.components_[j,k])
    ax.text(s_*pcaUS.components_[i,k],
    s_*pcaUS.components_[j,k],
    USArrests.columns[k])

print(scores.std(0, ddof =1))
print(pcaUS.explained_variance_)
print(pcaUS.explained_variance_ratio_)

#%% capture
fig , axes = plt.subplots (1, 2, figsize =(15, 6))
ticks = np.arange(pcaUS.n_components_)+1
ax = axes [0]

ax.plot(ticks,
        np.cumsum(pcaUS.explained_variance_ratio_),
        marker='o',
        color='orange')
ax.set_xlabel('Principal Component')
ax.set_ylabel('Cumulative Prop. Variance Explained')
ax.set_ylim([0, 1.1])
ax.set_xticks(ticks)
plt.tight_layout()
plt.show()

ax.plot(ticks ,
pcaUS.explained_variance_ratio_ ,
marker='o')
ax.set_xlabel('Principal Component ');
ax.set_ylabel('Proportion of Variance Explained ')
ax.set_ylim ([0 ,1])
ax.set_xticks(ticks)

a = np.array ([1,2,8,-3])
print(np.cumsum(a))

