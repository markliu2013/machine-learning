from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

X, y = load_iris(return_X_y=True)

clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X, y)
prediction = clf.predict([[5.3, 3.9, 1.2, 0.1], [1.3, 1.9, 0.2, 0.1], [11.3, 1.9, 0.2, 0.1]])
print(prediction)




