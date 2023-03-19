# Import libraries
import pandas as pd
import numpy as np

from flask import Flask, request, url_for, redirect, render_template, jsonify
from pycaret.classification import load_model, predict_model

# Create the app
app = Flask(__name__)

# Load trained Pipeline
model = load_model('price_model')

# Select columns for modle prediction
cols = ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
        'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',
        'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',
        'touch_screen', 'wifi']


@app.route('/')
def home():
    """
    Render route of the main page
    """
    return render_template("home.html")

# Define predict function


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict function 
    This function will use the price_model to prdict the phone price rank value the following params

    :battery_power: Total energy a battery can store in one time measured in mAh
    :blue: Has bluetooth (0-no, 1-yes)
    :clock_speed: Speed of the microprocessor
    :dual_sim: Has dual sim support (0-no, 1-yes)
    :fc: Front Camera MP 
    :four_g: Has 4G or not (0-no, 1-yes)
    :int_memory: Internal Memory GB
    :m_dep: Mobile Depth in cm
    :mobile_wt: Weight of mobile phone
    :n_cores: Number of cores of processor
    :pc: Primary Camera MP
    :px_height: Pixel Resolution Height
    :px_width: Pixel Resolution Width
    :ram: RAM in MB
    :sc_h: Screen Height of mobile in cm
    :sc_w: Screen Width of mobile in cm
    :talk_time: Longest time that a single battery charge will last when you are talking
    :three_g: Has 3G (0-no, 1-yes)
    :touch_screen: Has touch screen (0-no, 1-yes)
    :wifi: Has wifi (0-no, 1-yes)
    :return: Return the phone price rank prediction (0-3)
    """
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns=cols)
    prediction = predict_model(model, data=data_unseen, round=0)
    prediction = int(prediction.Label[0])
    return render_template('home.html', pred='Expected price rank will be {}'.format(prediction))


@app.route('/predict_api', methods=['POST'])
def predict_api():
    """
    Predict function 
    This function will use the price_model to prdict the phone price rank value the following params

    :battery_power: Total energy a battery can store in one time measured in mAh
    :blue: Has bluetooth (0-no, 1-yes)
    :clock_speed: Speed of the microprocessor
    :dual_sim: Has dual sim support (0-no, 1-yes)
    :fc: Front Camera MP 
    :four_g: Has 4G or not (0-no, 1-yes)
    :int_memory: Internal Memory GB
    :m_dep: Mobile Depth in cm
    :mobile_wt: Weight of mobile phone
    :n_cores: Number of cores of processor
    :pc: Primary Camera MP
    :px_height: Pixel Resolution Height
    :px_width: Pixel Resolution Width
    :ram: RAM in MB
    :sc_h: Screen Height of mobile in cm
    :sc_w: Screen Width of mobile in cm
    :talk_time: Longest time that a single battery charge will last when you are talking
    :three_g: Has 3G (0-no, 1-yes)
    :touch_screen: Has touch screen (0-no, 1-yes)
    :wifi: Has wifi (0-no, 1-yes)
    :return: Return the phone price rank prediction (0-3)
    """
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(model, data=data_unseen)
    output = prediction.Label[0]
    return jsonify(output)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
