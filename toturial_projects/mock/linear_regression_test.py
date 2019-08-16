import numpy as np

def test1():
    X1 = np.array([[1], [2], [3], [4], [5]])
    y1 = np.array([4, 7, 10, 13, 16])

    X2 = np.array([[1], [2], [3], [4], [5]])
    y2 = np.array([2, 4, 6, 8, 10])

    X3 = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y3 = np.dot(X3, np.array([1, 2])) + 3

    reg = LinearRegression().fit(X1, y1)
    print(reg.predict(np.array([[9], [10]])))

    reg = reg.fit(X2, y2)
    print(reg.predict(np.array([[9], [10]])))

    reg = reg.fit(X3, y3)
    print(reg.predict(np.array([[3, 5]])))

def test2():
    import pandas as pd
    # from sklearn.linear_model import LinearRegression
    from linear_regression import LinearRegression

    data_path = '../mnist/data/'
    df_train = pd.read_csv(data_path + "train.csv")
    X = df_train.iloc[:, 1:].to_numpy()
    y = df_train.iloc[:, 0].to_numpy()
    lg = LinearRegression().fit(X, y)

    df_test = pd.read_csv(data_path + "test.csv")
    predictions = lg.predict(df_test)
    submission = pd.DataFrame({
        "ImageId": range(1, 1 + len(predictions)),
        "Label": list(map(lambda x: int(round(x)), predictions))
    })
    submission.to_csv("mnist-submission9.csv", index=False)

test1()

