import flask
from flask import Flask, Response, jsonify, request
import pandas as pd
from io import StringIO
import joblib
#import csv

app = Flask(__name__)

clf=joblib.load(filename='classifier')

def run_model(input_arr):
    """Predictor function."""
    txfm_input_arr = input_arr[['Age','Fare']]
    pred = clf.predict(txfm_input_arr)
    #return 1
    return pd.DataFrame(clf.predict(txfm_input_arr),columns=['Predictions'])

@app.route('/ping', methods=['GET'])
def ping():
    """
    Determine if the container is healthy by running a sample through the algorithm.
    """
    # we will return status ok if the model doesn't barf
    # but you can also insert slightly more sophisticated tests here
    test_data = pd.read_csv("test.csv")
    try:
        result = run_model(test_data)
        return Response(response='{"status": "SUCCESS!"}', status=501, mimetype='application/json')
    except:
        return Response(response='{"status": "error"}', status=500, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def predict():
    """
    Do an inference on a single batch of data.
    curl -H "Content-Type: multipart/form-data" -F "file=@train.csv" http://0.0.0.0:8080/invocations
    """

    # Input file
    flask_file = request.files['file']
    if not flask_file:
        return 'Upload a CSV file'
    
    X_train = pd.read_csv(request.files.get('file'))
    results = run_model(X_train)
    return results.to_json(orient='values')
    