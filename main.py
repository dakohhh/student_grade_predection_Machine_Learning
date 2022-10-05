import pickle
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle 



def calculate_predictied_score(slope, hours, intercept):
    predicted_score =slope * hours + intercept

    return predicted_score




data = pd.read_csv("Student Study Hour V2.csv")

data_to_predict = "Scores"


print(data.head())
print()


#Prints out the Liner Corellation between Variables
#print(data.corr())


data_to_predict = "Scores"
x = np.array(data.drop([data_to_predict], 1))
y = np.array(data[data_to_predict])

#print(x, y)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

best_accuracy = 0
for _ in range(50):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    accuracy = linear.score(x_test, y_test)

    print("accuracy", accuracy)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        with open("student_data_model.pickle", "wb") as f:
            pickle.dump(linear, f)

pickle_in = open("student_data_model.pickle", "rb")



linear = pickle.load(pickle_in)

slope_or_coef = linear.coef_

intercept = linear.intercept_


predictions = linear.predict(x_test)



for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])



