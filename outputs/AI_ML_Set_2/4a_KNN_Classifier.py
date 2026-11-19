def main():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

    csv = Path(__file__).with_name("purchase_dataset.csv")
    if csv.exists():
        df = pd.read_csv(csv)
        X, y = df.iloc[:, [2, 3]].values, df.iloc[:, 4].values
        print("Dataset: purchase_dataset.csv")
    else:
        sample, y = make_classification(
            n_samples=400, n_features=2, n_redundant=0,
            n_clusters_per_class=1, random_state=42
        )
        X = np.column_stack((
            np.clip(sample[:, 0] * 12 + 40, 18, 70),
            np.clip(sample[:, 1] * 25000 + 60000, 15000, 150000)
        ))
        print("Dataset: generated sample (purchase_dataset.csv not found)")

    y = LabelEncoder().fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )
    scaler = StandardScaler()
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(scaler.fit_transform(X_train), y_train)
    y_pred = model.predict(scaler.transform(X_test))

    print("Predicted Values:", y_pred)
    print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("KNN Confusion Matrix")
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, data, labels, title in [
        (axes[0], X_train, y_train, "Training Set"),
        (axes[1], X_test, y_test, "Testing Set")
    ]:
        ax.scatter(data[:, 0], data[:, 1], c=labels, cmap="coolwarm", s=20)
        ax.set_title(title)
        ax.set_xlabel("Age")
        ax.set_ylabel("Annual Salary")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
