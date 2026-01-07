from flask import Flask, render_template, request
import numpy as np
import joblib
import math

app = Flask(__name__)

# Load trained model
loaded_models = joblib.load('models/my_trained_models.pkl')
model = loaded_models['xgboost']


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            # Collect inputs safely
            distance_to_solar_noon = float(request.form.get("distance_to_solar_noon"))
            temperature = float(request.form.get("temperature"))
            wind_speed = float(request.form.get("wind_speed"))
            sky_cover = int(request.form.get("sky_cover"))
            humidity = float(request.form.get("humidity"))
            avg_wind_speed = float(request.form.get("avg_wind_speed"))
            avg_pressure = float(request.form.get("avg_pressure"))
            wind_direction = int(request.form.get("wind_direction"))

            # Convert wind direction (1–36 → degrees)
            wind_deg = (wind_direction - 1) * 10

            # Cyclic encoding
            wind_dir_sin = math.sin(math.radians(wind_deg))
            wind_dir_cos = math.cos(math.radians(wind_deg))

            # Feature order MUST match training
            features = np.array([[ 
                distance_to_solar_noon,
                temperature,
                wind_speed,
                sky_cover,
                humidity,
                avg_wind_speed,
                avg_pressure,
                wind_dir_sin,
                wind_dir_cos
            ]])

            # Predict
            prediction = float(model.predict(features)[0])

        except Exception as e:
            print("Prediction error:", e)
            prediction = None

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
