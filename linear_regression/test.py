import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt
import pickle
from matplotlib import style


data = pd.read_csv("student_mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# print(data.head())

# LABEL
predict = "G3"

X = np.array(data.drop(columns=predict, axis=1))
Y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

# ------- UNCOMMENT TO TRAIN THE MODEL
"""best = 0
for _ in range(60):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

    linear = linear_model.LinearRegression()

    # find the best fit line
    linear.fit(x_train, y_train)

    # accuracy
    acc = linear.score(x_test, y_test)

    if acc > best:
        best = acc
        print("Accuracy: ", acc)
        # saving the model
        with open("student_model.pickle", "wb") as f:
            pickle.dump(linear, f)"""


pickle_in = open("student_model.pickle", "rb")
linear = pickle.load(pickle_in)

print("Coefficient: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

predictions = linear.predict(x_test)

print("Predictions: ")
for x in range (len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

p = "G1"
style.use("ggplot")
plt.scatter(data[p], data["G3"])
plt.xlabel(p)
plt.ylabel("Final Grade")
plt.show()
