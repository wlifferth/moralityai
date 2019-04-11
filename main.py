import json
import logging
import pickle
import pyrebase
import sys

from collections import defaultdict
from flask import Flask, render_template, request, jsonify
from google.cloud import automl_v1beta1
from google.cloud.automl_v1beta1.proto import service_pb2

from Sample import Sample

app = Flask(__name__)

with open("credentials/pyrebase.json") as pyrebase_config_file:
    pyrebase_config_json = json.load(pyrebase_config_file)
    firebase = pyrebase.initialize_app(pyrebase_config_json)
    db = firebase.database()

def predict(text):
    project_id = "moralityai-std"
    model_id = "TCN8676361882262876398"
    prediction_client = automl_v1beta1.PredictionServiceClient()
    name = 'projects/{}/locations/us-central1/models/{}'.format(project_id, model_id)
    payload = {'text_snippet': {'content': text, 'mime_type': 'text/plain' }}
    params = dict()
    prediction = prediction_client.predict(name, payload, params)
    if prediction.payload[0].display_name == "1":
        return float(prediction.payload[0].classification.score)
    else:
        return float(1 - prediction.payload[0].classification.score)

def push_sample_to_firebase(sample):
    result = db.child("samples").push(sample.get_firebase_dict())
    return result["name"]

@app.route("/", methods=["GET", "POST"])
def home():
    context = defaultdict(lambda: "")
    if request.method == "POST":
        sample = Sample(request.form["text"], prediction_score=predict(request.form["text"]))
        context["sample"] = sample
    return render_template("home.html", context=context)

@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500

@app.route("/_prediction_feedback")
def register_prediction_feedback():
    try:
        sample_text = request.args.get("sample_text", None, type=str)
        sample_prediction = request.args.get("sample_prediction", None, type=float)
        sample = Sample(sample_text, prediction_score=sample_prediction)
        correct = request.args.get("correct", None, type=bool)
    except:
        return jsonify(success=False)
    else:
        if correct:
            if sample_prediction > 0.5:
                sample.label = 1
            else:
                sample.label = 0
        else:
            if sample_prediction > 0.5:
                sample.label = 0
            else:
                sample.label = 1
        sample.labeled = True
        push_sample_to_firebase(sample)
        return jsonify(success=True)

if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
