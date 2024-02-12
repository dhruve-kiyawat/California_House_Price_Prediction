import pickle 
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd


app = Flask(__name__)
# Loading the pickeled model
model = pickle.load(open("linear_model.pkl", "rb"))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods = ['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    data = np.array(list(data.values())).reshape(1, -1)
    scaled_data = scaler.transform(data)
    output = model.predict(scaled_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict', methods = ['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    input = scaler.transform(np.array(data).reshape(1, -1))
    print(input)
    output = model.predict(input)[0]
    output = round(output, 2) * 100000
    return render_template("home.html", prediction_text = f"The House Price Prediction is {output} $")

if __name__=="__main__":
    app.run(debug = True)