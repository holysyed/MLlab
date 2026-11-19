def main():
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs

    X, y = make_blobs(
        n_samples=200, centers=3,
        cluster_std=1.0, random_state=42
    )
    model = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    centroids = model.cluster_centers_

    for i in range(3):
        print(f"\nCluster {i + 1}")
        print("Number of Points:", sum(labels == i))
        print("Centroid:", centroids[i])

    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=50)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                color="red", marker="X", s=200, label="Centroids")
    plt.title("K-Means Clustering")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
