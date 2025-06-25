
import pandas as pd
import joblib
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("model/final_dataset.csv")

# Drop rows with missing values
df.dropna(inplace=True)

# Define features and targets
features = df.drop(columns=["Year", "Bananas", "Cassava, fresh", "Coffee, green", "Tea leaves"])
targets = df[["Bananas", "Cassava, fresh", "Coffee, green", "Tea leaves"]]

# Train model
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model/model.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        input_data = {key: float(request.form[key]) for key in request.form}
        input_df = pd.DataFrame([input_data])
        model = joblib.load("model/model.pkl")
        prediction = model.predict(input_df)[0]
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
