def main():
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


if __name__ == "__main__":
    main()
