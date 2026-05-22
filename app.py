
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Adult Income Prediction API")
model = joblib.load("model.pkl")

class PredictionInput(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    educational_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    gender: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

@app.get("/")
def home():
    return {"message": "Adult Income Prediction API Running"}

@app.post("/predict")
def predict(data: PredictionInput):
    df = pd.DataFrame([{
        "age": data.age,
        "workclass": data.workclass,
        "fnlwgt": data.fnlwgt,
        "education": data.education,
        "educational-num": data.educational_num,
        "marital-status": data.marital_status,
        "occupation": data.occupation,
        "relationship": data.relationship,
        "race": data.race,
        "gender": data.gender,
        "capital-gain": data.capital_gain,
        "capital-loss": data.capital_loss,
        "hours-per-week": data.hours_per_week,
        "native-country": data.native_country
    }])
    prediction = model.predict(df)[0]
    return {"prediction": str(prediction)}
