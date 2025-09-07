from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model and vectorizer
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    message = request.form["message"]

    # Transform message using vectorizer
    data = vectorizer.transform([message]).toarray()

    # Predict using model
    prediction = model.predict(data)[0]
    result = "Spam" if prediction == 1 else "Not Spam"

    return render_template("index.html", prediction_text=f"The message is: {result}")

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
