from flask import Flask, request, abort, render_template
import json
import pickle
import config.predict

app = Flask(__name__)
dt_model=pickle.load(open('./config/dt_model.pkl','rb'))
mlp_model=pickle.load(open('./config/mlp_model.pkl','rb'))
knn_model=pickle.load(open('./config/knn_model.pkl','rb'))
nb_model=pickle.load(open('./config/nb_model.pkl','rb'))
rfc_model=pickle.load(open('./config/rfc_model.pkl','rb'))