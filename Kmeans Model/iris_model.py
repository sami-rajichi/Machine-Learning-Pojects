import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
iris = datasets.load_iris()
dataset = pd.DataFrame(iris.data)
dataset.columns = ['height_sepale', 'width_sepale', 'height_petale', 'width_petale']
print(dataset.head())

from sklearn.cluster import KMeans

clusters = []
for i in range(1,16):
  km = KMeans(n_clusters=i)
  km.fit(dataset)
  clusters.append(km.inertia_)
  
plt.plot(range(1,16), clusters)
plt.title('Eblow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia inter-classes')
print(plt.show())

km = KMeans(n_clusters=3)
km.fit(dataset)

colors = np.array(['red','green','blue'])
plt.scatter(dataset.height_petale, dataset.width_petale, c=colors[km.labels_], s=20)
#plt.scatter(dataset.height_petale, dataset.height_petale, c=colors[iris.target], s=20)