from flask import Flask,request
import pandas as pd
import numpy as np
import pickle
import joblib

app=Flask(__name__)
pickle_in=open('knn.pkl','rb')
knn=pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "HELLO"

@app.route('/predict')
def predict():
    variance=request.args.get("variance")
    skewness=request.args.get("skewness")
    curtosis=request.args.get("curtosis")
    entropy=request.args.get("entropy")
    prediction=knn.predict([[variance,skewness,curtosis,entropy]])
    if prediction==0:
        return "Unauthorized note"
    else:
        return "Authentic note"

@app.route('/predict_file',methods=["POST"])
def predict_file():
    df_test=pd.read_csv(request.files.get("file"))
    # print(df_test.head())
    prediction=knn.predict(df_test)
    return str(list(prediction))

    
if __name__=='__main__':
    app.run()