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

data = pd.read_csv("./project/config/Thai_rain_clean.csv")

x = pd.DataFrame(data,columns=['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',
       'Sunshine', 'WindGustSpeed', 'WindSpeed9am','WindSpeed3pm', 'Humidity9am', 
       'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm', 'RainToday'])
y = pd.DataFrame(data,columns=['RainTomorrow'])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std  = sc.transform(x_test)

#NeuralNetwork
clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(10, ), random_state=0)
clf.fit(x_train_std, y_train)

#RandomForest
rfc = RandomForestClassifier(n_estimators=15, max_depth=None, min_samples_split=2, random_state=0)
rfc.fit(x_train_std, y_train)
# rfc_predict = rfc.predict(x_test)
# print(rfc_predict)
# print("dd")

# #DecisionTree
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train_std, y_train)

#Knn
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train_std, y_train)

#NaiveBayes
gnb = GaussianNB()
gnb.fit(x_train_std, y_train)


pickle.dump(clf,open('./project/config/clf_model.pkl','wb'))
pickle.dump(rfc,open('./project/config/rfc_model.pkl','wb'))
pickle.dump(dt,open('./project/config/dt_model.pkl','wb'))
pickle.dump(knn,open('./project/config/knn_model.pkl','wb'))
pickle.dump(gnb,open('./project/config/gnb_model.pkl','wb'))