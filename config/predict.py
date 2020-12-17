import numpy as np
import pandas as pd
import warnings
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

warnings.filterwarnings("ignore")

data = pd.read_csv("churn.csv")

x = pd.DataFrame(data,columns=['Cloud Type', 'Precip Type', 'Temperature (C)', 'Apparent Temperature (C)',
       'Humidity', 'Wind Speed (km/h)', 'Wind Bearing (degrees)',
       'Visibility (km)', 'Pressure (millibars)'])
y = pd.DataFrame(data,columns=['Daily Summary'])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std  = sc.transform(x_test)

#NeuralNetwork
clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(10, ), random_state=1)
clf.fit(x_train_std, y_train)

#RandomForest
rfc = RandomForestClassifier(n_estimators=15, max_depth=None, min_samples_split=2, random_state=0)
rfc.fit(x_train_std, y_train)
rfc_predict = rfc.predict(x_test)
print(rfc_predict)
print("dd")

# #DecisionTree
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train_std, y_train)

#Knn
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train_std, y_train)

#NaiveBayes
gnb = GaussianNB()
gnb.fit(x_train_std, y_train)


pickle.dump(clf,open('clf_model.pkl','wb'))
pickle.dump(rfc,open('rfc_model.pkl','wb'))
pickle.dump(dt,open('dt_model.pkl','wb'))
pickle.dump(knn,open('knn_model.pkl','wb'))
pickle.dump(gnb,open('gnb_model.pkl','wb'))