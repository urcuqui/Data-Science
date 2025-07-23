from flask import Flask, render_template, request, jsonify
import joblib
app = Flask(__name__)


import os
model_filename = os.path.join(os.path.dirname(__file__), 'full.joblib')

loaded_model = joblib.load(model_filename)

def classify_message(text):    
    predictions = loaded_model.predict([text])
    if predictions[0] == 0:
        return "0-250000"
    elif predictions[0] == 1:
        return "250000-350000"
    elif predictions[0] == 2:
        return "350000-450000"
    elif predictions[0] == 3:
        return "450000-550000"
    else:
        return "650000+"


if __name__ == "__main__":
    app.run(debug=False, port=5001)