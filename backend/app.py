import os
import pandas as pd
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dataprocessing.TS_analysis import analyze_time_series
from io import StringIO
from ml2 import run_forecast

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/root", methods=["GET"])
def home():
    return jsonify({"message": "Flask API is running!"})

@app.route("/upload", methods=["POST"])
def upload_file():
    # Data retrieval via file upload
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        # Read the CSV directly from the uploaded file
        df = pd.read_csv(file)
        result = analyze_time_series(df)
        # Preprocessing step will save uploads/processed_data.csv;
        # the /forecast endpoint can then train XGBoost and generate forecasts.
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/fetch", methods=["POST"])
def fetch_data():
    # Data retrieval via URL/API
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "No URL provided"}), 400
    url = data["url"]

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        result = analyze_time_series(df)
        return jsonify(result), 200
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to fetch data from URL: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/forecast", methods=["POST"])
def forecast():
    """
    Trigger XGBoost-based time series forecasting using the preprocessed data.
    Expects that `/upload` or `/fetch` has already been called successfully.
    """
    try:
        result = run_forecast()
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/forecast/plot", methods=["GET"])
def forecast_plot():
    """
    Serve the Matplotlib-generated forecast plot PNG file so the React
    dashboard can embed it directly.
    """
    filename = "forecast_plot.png"
    plot_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(plot_path):
        return jsonify({"error": "Forecast plot not found. Run /forecast first."}), 404
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
