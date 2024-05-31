from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

app = Flask(__name__)
CORS(app)

@app.route("/api/predict/", methods=['GET'])
def predict():

    dayCharge = float(request.args['dayCharge'])
    eveCharge = float(request.args['eveCharge'])
    nightCharge = float(request.args['nightCharge'])
    intlCharge = float(request.args['intlCharge'])

    dayCalls = int(request.args['dayCalls'])
    eveCalls = int(request.args['eveCalls'])
    nightCalls = int(request.args['nightCalls'])
    intlCalls = int(request.args['intlCalls'])

    dayMinutes = float(request.args['dayMinutes'])
    eveMinutes = float(request.args['eveMinutes'])
    nightMinutes = float(request.args['nightMinutes'])
    intlMinutes = float(request.args['intlMinutes'])

    customerServiceCalls = int(request.args['customerServiceCalls'])
    internationalPlan = int(request.args['internationalPlan'])
    numberVmailMessages = int(request.args['numberVmailMessages'])


    total_call_duration = dayMinutes + eveMinutes + nightMinutes + intlMinutes

    if (dayCalls + eveCalls + nightCalls + intlCalls) != 0:
        avg_call_duration = total_call_duration / (dayCalls + eveCalls + nightCalls + intlCalls)
    else:
        avg_call_duration = 0  

    total_call_charge = dayCharge + eveCharge + nightCharge + intlCharge

    if total_call_duration != 0:
        cost_per_minute = total_call_charge / total_call_duration
    else:
        cost_per_minute = 0  

    input_features = [
        total_call_charge,
        internationalPlan,
        customerServiceCalls,
        dayMinutes,
        total_call_duration,
        dayCharge,
        intlCalls,
        numberVmailMessages,
        intlCharge,
        intlMinutes,
        cost_per_minute,
        avg_call_duration
    ]


    rfc_model = RandomForestClassifier(random_state=0)
    model_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RandomForestClassification.pkl")
    with open(model_filename, 'rb') as f:
        rfc_model = pickle.load(f)

    prediction = rfc_model.predict([input_features])
    prediction_proba = rfc_model.predict_proba([input_features])

    return jsonify({
        'prediction': int(prediction[0]),
        'probability': {
            'not_churn': round(prediction_proba[0][0], 4),
            'churn': round(prediction_proba[0][1], 4)
        }
    })


@app.route("/")
def hello_world():

    print(request.args)

    return "<p>Hello, World!</p>"
    

