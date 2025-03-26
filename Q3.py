import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load dataset
url = "https://raw.githubusercontent.com/selva86/datasets/master/supermarket_sales.csv"
df = pd.read_csv(url)

# Select relevant features (Purchase History & Demographics)
df = df[['Total', 'Quantity', 'Unit price', 'gross income']]  

# Normalize data
scaler = StandardScaler()
X = scaler.fit_transform(df)

################################# CELL 2 ##########################
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.show()

############################### CELL 3 #########################
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X)

############################# CELL 4 ###########################
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Cluster'], cmap='viridis', alpha=0.6)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Customer Clusters')
plt.colorbar(label="Cluster")
plt.show()
