"""AI & ML Lab Set 2: choose an experiment from 4(a) to 6(b).
Run: python AI_ML_Set_2_All_Experiments.py
Or:  python AI_ML_Set_2_All_Experiments.py 5b
"""
def experiment_4a():
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

def experiment_4b():
    import random
    import matplotlib.pyplot as plt

    random.seed(7)
    room = [[random.choice([0, 1]) for _ in range(3)] for _ in range(3)]

    class StateAgent:
        def __init__(self):
            self.x = self.y = 0
            self.battery = 100
            self.state = "SEARCHING"
            self.score = 0

        def step(self):
            if self.state == "SEARCHING":
                if self.battery <= 20:
                    self.state = "RETURNING"
                elif room[self.x][self.y]:
                    self.state = "SUCKING"
                else:
                    self.x, self.y = random.randrange(3), random.randrange(3)
                    self.battery -= 5
            elif self.state == "SUCKING":
                room[self.x][self.y] = 0
                self.battery -= 10
                self.score += 10
                self.state = "SEARCHING"
            elif self.state == "RETURNING":
                self.x = self.y = 0
                self.battery -= 5
                self.state = "CHARGING"
            elif self.state == "CHARGING":
                self.battery = min(100, self.battery + 25)
                if self.battery == 100:
                    self.state = "SEARCHING"

    robot = StateAgent()
    for i in range(1, 9):
        if random.random() < 0.3:
            room[random.randrange(3)][random.randrange(3)] = 1
        robot.step()
        print(f"Step {i}: Position=({robot.x},{robot.y}), "
              f"State={robot.state}, Battery={robot.battery}%")
    print("Final Score:", robot.score)

    plt.imshow(room, cmap="RdYlGn_r", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            plt.text(j, i, "Dirty" if room[i][j] else "Clean",
                     ha="center", va="center", fontsize=9)
    plt.scatter(robot.y, robot.x, marker="*", s=250, color="blue")
    plt.xticks(range(3))
    plt.yticks(range(3))
    plt.title("State-Based Agent - Final Room Status")
    plt.show()

def experiment_5a():
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.naive_bayes import GaussianNB
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

    data = {
        "Age": [22, 25, 47, 52, 46, 56, 23, 24, 48, 50],
        "Salary": [20000, 25000, 50000, 60000, 52000,
                   65000, 22000, 24000, 55000, 58000],
        "Purchased": [0, 0, 1, 1, 1, 1, 0, 0, 1, 1]
    }
    df = pd.DataFrame(data)
    X, y = df[["Age", "Salary"]], df["Purchased"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Actual:", y_test.values)
    print("Predicted:", y_pred)
    print("Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")
    print(classification_report(y_test, y_pred, zero_division=0))
    customer = pd.DataFrame([[30, 30000]], columns=["Age", "Salary"])
    print("New Customer Prediction:", model.predict(customer)[0])

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Naive Bayes Confusion Matrix")
    plt.show()

def experiment_5b():
    import heapq
    import matplotlib.pyplot as plt

    graph = {
        "A": {"B": 2, "C": 5},
        "B": {"D": 2, "E": 4},
        "C": {"G": 6},
        "D": {"G": 4},
        "E": {"G": 1},
        "G": {}
    }

    def ucs(start, goal):
        queue = [(0, start, [start])]
        visited = set()
        while queue:
            cost, node, path = heapq.heappop(queue)
            if node == goal:
                return cost, path
            if node in visited:
                continue
            visited.add(node)
            for neighbour, weight in graph[node].items():
                if neighbour not in visited:
                    heapq.heappush(queue, (
                        cost + weight, neighbour, path + [neighbour]
                    ))
        return None

    result = ucs("A", "G")
    if result is None:
        print("Path not found")
        return
    cost, path = result
    print("Minimum Cost:", cost)
    print("Optimal Path:", " -> ".join(path))

    pos = {
        "A": (0, 1), "B": (1, 2), "C": (1, 0),
        "D": (2, 2), "E": (2, 1), "G": (3, 1)
    }
    chosen_edges = set(zip(path, path[1:]))
    for a, neighbours in graph.items():
        for b, weight in neighbours.items():
            x1, y1 = pos[a]
            x2, y2 = pos[b]
            chosen = (a, b) in chosen_edges
            plt.plot([x1, x2], [y1, y2],
                     color="green" if chosen else "grey",
                     linewidth=3 if chosen else 1)
            plt.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, str(weight))
    for node, (x, y) in pos.items():
        plt.scatter(x, y, s=600, color="steelblue")
        plt.text(x, y, node, ha="center", va="center", color="white")
    plt.title("Uniform Cost Search - Optimal Path")
    plt.axis("off")
    plt.show()

def experiment_6a():
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

def experiment_6b():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

    X = np.arange(1, 11).reshape(-1, 1)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Actual Values:", y_test)
    print("Predicted Values:", y_pred)
    print("Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Prediction for 6 hours:", model.predict([[6]])[0])
    print("Probability:", model.predict_proba([[6]]))

    plt.figure(figsize=(7, 4))
    plt.scatter(X, y, color="blue", label="Actual Data")
    x_curve = np.linspace(1, 10, 100).reshape(-1, 1)
    plt.plot(x_curve, model.predict_proba(x_curve)[:, 1],
             color="red", label="Logistic Regression")
    plt.xlabel("Hours")
    plt.ylabel("Probability")
    plt.title("Logistic Regression")
    plt.legend()
    plt.grid(True)
    plt.show()

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    import sys
    choices = {
        "4a": ("KNN Classifier", experiment_4a),
        "4b": ("State-Based Agent", experiment_4b),
        "5a": ("Naive Bayes Classifier", experiment_5a),
        "5b": ("Uniform Cost Search", experiment_5b),
        "6a": ("K-Means Clustering", experiment_6a),
        "6b": ("Supervised Case Study: Logistic Regression", experiment_6b)
    }
    if len(sys.argv) > 1:
        choice = sys.argv[1].strip().lower().replace("(", "").replace(")", "")
    else:
        print("AI & ML Lab — Set 2")
        for key, (name, _) in choices.items():
            print(f"{key}: {name}")
        choice = input("Choose experiment (4a/4b/5a/5b/6a/6b): ").strip().lower()
        choice = choice.replace("(", "").replace(")", "")
    if choice not in choices:
        print("Invalid choice. Use 4a, 4b, 5a, 5b, 6a or 6b.")
        sys.exit(1)
    print("\nRunning", choices[choice][0], "\n")
    choices[choice][1]()
