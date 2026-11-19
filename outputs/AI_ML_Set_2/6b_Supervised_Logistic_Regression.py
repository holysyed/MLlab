def main():
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
    main()
