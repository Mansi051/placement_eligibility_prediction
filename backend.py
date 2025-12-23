from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from model import model,scaler

app=FastAPI(
    title="Placement Eligibility API",
    description="Predict placement eligibility using cgpa and backlogs",
    version="1.0"
)

class StudentInput(BaseModel):
    cgpa:float
    backlogs:int

@app.get("/")
def home():
    return {"message":"Placement Eligibility API is running"}

@app.post("/predict")
def predict_eligibility(data:StudentInput):
    x_input=np.array([[data.cgpa,data.backlogs]])
    x_scaled=scaler.transform(x_input)

    probability=model.predict_proba(x_scaled)[0][1]
    prediction=model.predict(x_scaled)[0]

    return{
        "cgpa": data.cgpa,
        "backlogs":data.backlogs,
        "probability":round(probability,2),
        "eligibility":"Eligible" if prediction==1 else "Not Eligible"
    }

