import pandas as pd
import joblib
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import os

# Load and clean dataset
df = pd.read_csv("model/final_dataset.csv")
df.dropna(inplace=True)

# Define target and feature columns
target_cols = ["Bananas", "Cassava, fresh", "Coffee, green", "Tea leaves"]
feature_cols = [col for col in df.columns if col not in ["Year"] + target_cols]

X = df[feature_cols]
y = df[target_cols]

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model and feature list
joblib.dump(model, "model/model.pkl")
joblib.dump(feature_cols, "model/feature_columns.pkl")

# Flask App
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    feature_columns = joblib.load("model/feature_columns.pkl")

    if request.method == "POST":
        input_data = {key: float(request.form[key]) for key in feature_columns}
        input_df = pd.DataFrame([input_data])
        model = joblib.load("model/model.pkl")
        prediction = model.predict(input_df)[0]

    return render_template(
        "index.html", feature_columns=feature_columns, prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)
