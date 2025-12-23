from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from model import model,scaler
import joblib

app=FastAPI()

model=joblib.load('placement_model.pkl')
scaler=joblib.load('scaler.pkl')

class StudentInput(BaseModel):
    cgpa:float
    backlogs:int

@app.post("/predict")
def predict_eligibility(data:StudentInput):
    x_input=np.array([[data.cgpa,data.backlogs]])
    x_scaled=scaler.transform(x_input)

    prediction=model.predict(x_scaled)[0]

    return{
        "cgpa": data.cgpa,
        "backlogs":data.backlogs,
        "eligibility":"Eligible" if prediction==1 else "Not Eligible"
    }

