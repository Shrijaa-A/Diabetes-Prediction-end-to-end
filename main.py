import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np


app = Flask(__name__)
data = pd.read_csv('diabetes.csv')
pipe = pickle.load(open('model.pkl','rb'))

@app.route("/")
def index():
    # locations = sorted(data['details'].unique())
    return render_template('form.html')

@app.route('/predict', methods = ['POST'])
def predict():
    Pregnancies = float(request.form.get('Pregnancies'))
    GlucoseLevel = float(request.form.get('GlucoseLevel')) 
    BloodPressure = float(request.form.get('BloodPressure'))
    SkinThickness = float(request.form.get('SkinThickness'))
    Insulin = float(request.form.get('Insulin'))
    BMI = float(request.form.get('BMI'))
    DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
    Age = float(request.form.get('Age'))


    print(Pregnancies, GlucoseLevel, BloodPressure, SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age)
    input = np.array([[Pregnancies, GlucoseLevel, BloodPressure, SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
    print(input)
   # prediction = pipe.predict(input)[0]
    proba = pipe.predict_proba(input)[0]
    print(proba)
    print(np.round(proba[1],5))
    return str(np.round(proba[1]*100,5))

if __name__ == '__main__':
    app.run(debug = True, host = "0.0.0.0", port = 9696)