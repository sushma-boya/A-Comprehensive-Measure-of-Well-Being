from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model
with open("models/hdi_model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Get user inputs
    life = float(request.form["life"])
    expected = float(request.form["expected"])
    mean = float(request.form["mean"])
    gni = float(request.form["gni"])

    # Create DataFrame with correct feature names
    features = pd.DataFrame({
        "Life expectancy at birth": [life],
        "Expected years of schooling": [expected],
        "Mean years of schooling": [mean],
        "Gross national income (GNI) per capita": [gni]
    })

    # Make prediction
    prediction = model.predict(features)[0]
    prediction = round(float(prediction), 3)

    percentage = round(prediction * 100, 1)

    return render_template(
        "result.html",
        prediction=prediction,
        percentage=percentage
    )


if __name__ == "__main__":
    app.run(debug=True)