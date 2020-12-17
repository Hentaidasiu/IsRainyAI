from flask import Flask, request, abort, render_template
import json
import pickle
import project.config.predict



app = Flask(__name__,template_folder='HTML')
clf_model=pickle.load(open('./project/config/clf_model.pkl','rb')) #NeuralNetwork
rfc_model=pickle.load(open('./project/config/rfc_model.pkl','rb')) #RandomForest
dt_model=pickle.load(open('./project/config/dt_model.pkl','rb')) #DecisionTree
knn_model=pickle.load(open('./project/config/knn_model.pkl','rb')) #Knn
gnb_model=pickle.load(open('./project/config/gnb_model.pkl','rb')) #NaiveBayes


@app.route('/')
def main():
    return render_template('landing2.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    rainy = 0
    clear = 0
    MinTemp = request.form['MinTemp']
    MaxTemp = request.form['MaxTemp']
    Rainfall = request.form['Rainfall']
    Evaporation = request.form['Evaporation']
    Sunshine = request.form['Sunshine']
    WindGustSpeed = request.form['WindGustSpeed']
    WindSpeed9am = request.form['WindSpeed9am']
    WindSpeed3pm = request.form['WindSpeed3pm']
    Humidity9am = request.form['Humidity9am']
    Humidity3pm = request.form['Humidity3pm']
    Pressure9am = request.form['Pressure9am']
    Pressure3pm = request.form['Pressure3pm']
    Temp9am = request.form['Temp9am']
    Temp3pm = request.form['Temp3pm']
    RainToday = request.form['RainToday']
    X=[float(MinTemp),
                float(MaxTemp),
                float(Rainfall),
                float(Evaporation),
                float(Sunshine),
                float(WindGustSpeed),
                float(WindSpeed9am),
                float(WindSpeed3pm),
                float(Humidity9am),
                float(Humidity3pm),
                float(Pressure9am),
                float(Pressure3pm),
                float(Temp9am),
                float(Temp3pm),
                float(RainToday)]
    predict_clf = clf_model.predict([X]) #NeuralNetwork
    predict_rfc = rfc_model.predict([X]) #RandomForest
    predict_dt = dt_model.predict([X]) #DecisionTree
    predict_knn = knn_model.predict([X]) #Knn
    predict_gnb = gnb_model.predict([X]) #NaiveBayes

    predict_list = [predict_clf,predict_rfc,predict_dt,predict_knn,predict_gnb]
    # f = open("./Project/Config/history.txt", "a")
    # f.write(str(features))
    # f.write("\n")
    # f.close()
    for x in predict_list:
        if x == [1]:
            rainy += 1
        elif x == [0]:
            clear +=1
        else:
            raise EnvironmentError
    if rainy > clear :
        print("Yes, Its rainy!")
        return render('result.html',pred='rainy')
    else :
        print("No, Its clear!")
        return render('result.html',pred='clear')
    
    return "ok"
