
from flask import Flask, request, render_template
import pandas as pd
import joblib   

app = Flask(__name__)

# Load trained model
model = joblib.load("car_price_model.pkl")   
model_columns = joblib.load("model_columns.pkl")   

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    manufacturer = request.form["Manufacturer"]
    model_name = request.form["Model"]
    fuel = request.form["Fuel type"]
    mileage = float(request.form["Mileage"])
    engine = float(request.form["Engine size"])
    year = int(request.form["Year"])

    data = {
        "Manufacturer": manufacturer,
        "Model": model_name,
        "Fuel type": fuel,
        "Mileage": mileage,
        "Engine size": engine,
        "Year of manufacture": year
    }

    df = pd.DataFrame([data])

    df = pd.get_dummies(df)
    df = df.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(df)[0]
    formatted_price = "₹{:,.0f}".format(prediction)

    return render_template(
        "index.html",
        prediction_text=f"Estimated Resale Price: {formatted_price}",
        manufacturer=manufacturer,
        model=model_name,
        fuel=fuel,
        mileage=mileage,
        engine=engine,
        year=year
    )

if __name__ == "__main__":
    app.run(debug=True)
