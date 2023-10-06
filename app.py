from flask import Flask, render_template,redirect,request
import pickle
import pandas as pd
import numpy as np
app = Flask(__name__)


app=Flask(__name__, template_folder='./templates', static_folder='./static')

rfr = pickle.load(open('DiabetesPrediction.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    features=[]
    preg = request.form.get("pregnancies")
    features.append(int(preg))
    gluc = request.form.get("glucose")
    features.append(int(gluc))
    bp   = request.form.get("bloodpressure")
    features.append(int(bp))
    skin = request.form.get("skinthickness")
    features.append(int(skin))
    insulin = request.form.get("insulin")
    features.append(int(insulin))
    bmi  = request.form.get("bmi")
    features.append(float(bmi))
    diapedf = request.form.get("diapedf")
    features.append(float(diapedf))
    age = request.form.get("age")
    features.append(int(age))
    final_features =np.array(features)
    final_features = final_features.reshape((1, 8))

    result = rfr.predict(final_features)
    if result[0] == 1:
        return render_template('result.html',result = "Person is diabetic")
    else:
        return render_template('result.html',result = "Person is non-diabetic")

if __name__=='__main__':
    app.run(debug=True)
