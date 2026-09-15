import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

data = pd.read_csv(r'C:\Users\POOJA\Downloads\income.csv')
print(data)
print(data.isnull().sum())
print(data.duplicated().sum())

scaler = MinMaxScaler()
features = data[['Age', 'Income($)']]
x_scaled = scaler.fit_transform(features)

inertia = []
for k in range(1,11):
    kmeans = KMeans(n_clusters = k, random_state = 42)
    kmeans.fit(x_scaled)
    inertia.append(kmeans.inertia_)
    
# plot
plt.figure(figsize = (6,4))
plt.plot(range(1,11), inertia, marker = 'o')
plt.title('elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()
#k =3

kmeans1 = KMeans(n_clusters = 3, random_state = 42)
clusters = kmeans1.fit_predict(x_scaled)
data['clusters'] = clusters

pca = PCA(n_components = 2)
x_pca = pca.fit_transform(x_scaled)
data['pca1'] = x_pca[:,0]
data['pca2'] = x_pca[:,1]
plt.figure(figsize = (6,4))
sns.scatterplot(x='pca1', y ='pca2', hue = 'clusters', data = data, palette ='Set1')
plt.title('KMeans Clustering with PCA')
plt.show()
