from flask import Flask, request, jsonify, render_template
import pickle
import re
import requests
import pandas as pd
import os

app = Flask(__name__)

# ===============================
# LOAD MODEL
# ===============================
model = pickle.load(open("sentiment_model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

DATASET_FILE = "dataset.csv"

# ===============================
# CREATE DATASET IF NOT EXISTS
# ===============================
if not os.path.exists(DATASET_FILE):

    df = pd.DataFrame(columns=[
        "Location",
        "Sentiment",
        "lat",
        "lon"
    ])

    df.to_csv(DATASET_FILE,index=False)

# ===============================
# TEXT PREPROCESS
# ===============================
def preprocess(text):

    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]"," ",text)

    return text


# ===============================
# HOME PAGE
# ===============================
@app.route("/")
def home():
    return render_template("index.html")


# ===============================
# PREDICT SENTIMENT
# ===============================
@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    text = data["text"]

    clean = preprocess(text)

    vec = vectorizer.transform([clean])

    prediction = model.predict(vec)[0]

    if prediction == "Negative":
        risk = 80
        emotion = "Fear"

    elif prediction == "Neutral":
        risk = 50
        emotion = "Neutral"

    else:
        risk = 20
        emotion = "Comfort"

    return jsonify({
        "sentiment": prediction,
        "emotion": emotion,
        "risk_score": risk
    })


# ===============================
# GET COORDINATES FOR NEW AREA
# ===============================
def get_coordinates(area):

    url = f"https://nominatim.openstreetmap.org/search?q={area}+Pune+India&format=json"

    response = requests.get(url).json()

    if len(response) > 0:

        lat = float(response[0]["lat"])
        lon = float(response[0]["lon"])

        return lat, lon

    return None, None


# ===============================
# SAVE FEEDBACK
# ===============================
@app.route("/submit-feedback", methods=["POST"])
def submit_feedback():

    data = request.get_json()

    area = data["area"]
    sentiment = data["sentiment"]

    df = pd.read_csv(DATASET_FILE)

    # check if area already exists
    if area not in df["Location"].values:

        lat, lon = get_coordinates(area)

    else:

        row = df[df["Location"] == area].iloc[0]
        lat = row["lat"]
        lon = row["lon"]

    new_row = {
        "Location": area,
        "Sentiment": sentiment,
        "lat": lat,
        "lon": lon
    }

    df = pd.concat([df, pd.DataFrame([new_row])])

    df.to_csv(DATASET_FILE,index=False)

    return jsonify({"status":"saved"})


# ===============================
# SEND MAP DATA
# ===============================
@app.route("/map-data")
def map_data():

    df = pd.read_csv(DATASET_FILE)

    score_map = {
        "Positive": 1,
        "Neutral": 0,
        "Negative": -1
    }

    df["score"] = df["Sentiment"].map(score_map)

    area_scores = df.groupby(["Location","lat","lon"])["score"].mean().reset_index()

    result = []

    for _, row in area_scores.iterrows():

        score = row["score"]

        if score > 0.3:
            color = "green"

        elif score < -0.3:
            color = "orange"

        else:
            color = "yellow"

        result.append({
            "area": row["Location"],
            "lat": row["lat"],
            "lng": row["lon"],
            "color": color
        })

    return jsonify(result)


# ===============================
# RUN APP
# ===============================
if __name__ == "__main__":
    app.run(debug=True)