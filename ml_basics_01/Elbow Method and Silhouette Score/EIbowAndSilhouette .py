import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

def elbow_and_silhouette():
    df = pd.read_csv("Mall_Customers.csv")

    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    inertia = []
    silhouette_scores = []

    K_range = range(2, 9)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=70)
        labels = kmeans.fit_predict(X_scaled)

        inertia.append(kmeans.inertia_)#Elbow plot
        silhouette_scores.append(silhouette_score(X_scaled, labels)) # Silhouette plot

    #Elbow plot
    plt.figure()
    plt.plot(K_range, inertia, marker='o')
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.show()

    # Silhouette plot
    plt.figure()
    plt.plot(K_range, silhouette_scores, marker='o')
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score vs K")
    plt.show()

elbow_and_silhouette()
