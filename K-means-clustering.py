import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
print("OPENING AND READING CSV")
data = pd.read_csv(r"C:\Users\gaurm\Downloads\mall-customers-data.csv", encoding='latin-1')
x = np.array(data["annual_income"])
y = np.array(data["spending_score"])
X = np.column_stack((x, y))
wcss = []
Y  = [1,2,3,4,5,6,7,8,9,10,11,12]
for i in range(1,13):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state= 42)  
    kmeans.fit(X)  
    wcss.append(kmeans.inertia_)  
plt.figure(figsize=(10, 6))
plt.scatter(Y, wcss, color='blue', alpha=0.7)
plt.show()
ideal_number_of_clusters = 5 #generated using elbow method
kmeans = KMeans(n_clusters=5, init='k-means++', random_state= 42)  
y_predict= kmeans.fit_predict(X)  
print(y_predict)
centroids = kmeans.cluster_centers_
plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green', 'purple', 'orange']  
for cluster in range(ideal_number_of_clusters):
   
    cluster_points = X[y_predict == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                s=50, color=colors[cluster], label=f'Cluster {cluster+1}', alpha=0.7)


plt.scatter(centroids[:, 0], centroids[:, 1], 
            s=200, color='black',label='Centroids')
plt.title("Customer Clusters with Centroids")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.show()


