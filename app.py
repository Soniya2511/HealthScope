from flask import Flask, render_template, request
import pickle
import numpy as np


app = Flask(__name__)


# Load trained models
diabetes_model = pickle.load(open("diabetes.pkl", "rb"))
heart_model = pickle.load(open("heart.pkl", "rb"))
parkinsons_model = pickle.load(open("parkinsons.pkl", "rb"))


# Home Page
@app.route("/")
def home():
    return render_template("home.html")


# Diabetes Page
@app.route("/diabetes")
def diabetes():
    return render_template("diabetes.html")


# Heart Disease Page
@app.route("/heartdisease")
def heartdisease():
    return render_template("heartdisease.html")


# Parkinson's Page
@app.route("/parkinsons")
def parkinsons():
    return render_template("parkinsons.html")


# Diabetes Prediction
@app.route("/predictdiabetes", methods=["POST"])
def predictdiabetes():

    features = [float(x) for x in request.form.values()]
    prediction = diabetes_model.predict([np.array(features)])

    if prediction[0] == 1:
        result = "High risk of Diabetes detected."
    else:
        result = "No significant Diabetes risk detected."

    return render_template(
        "diabetes.html",
        output_text=result
    )


# Heart Disease Prediction
@app.route("/predictheartdisease", methods=["POST"])
def predictheartdisease():

    features = [float(x) for x in request.form.values()]
    prediction = heart_model.predict([np.array(features)])

    if prediction[0] == 1:
        result = "High risk of Heart Disease detected."
    else:
        result = "No significant Heart Disease risk detected."

    return render_template(
        "heartdisease.html",
        output_text=result
    )


# Parkinson's Prediction
@app.route("/predictparkinson", methods=["POST"])
def predictparkinsons():

    features = [float(x) for x in request.form.values()]
    prediction = parkinsons_model.predict([np.array(features)])

    if prediction[0] == 1:
        result = "High risk of Parkinson’s Disease detected."
    else:
        result = "No significant Parkinson’s Disease risk detected."

    return render_template(
        "parkinsons.html",
        output_text=result
    )


# Run Application
if __name__ == "__main__":
    app.run(debug=True)