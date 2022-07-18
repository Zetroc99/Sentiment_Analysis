import pandas as pd
import pickle
from fastapi import FastAPI, Request
from pydantic import BaseModel

# Initialize API
api = FastAPI()

pkl_filename = 'model/airline_model.pkl'
with open(pkl_filename, 'rb') as file:
    lr_model = pickle.load(file)


# Data validation
class Airline(BaseModel):
    text: str


@api.post('/predict_simple')
async def predict_simple(request: Request):
    input_data = await request.json()
    input_df = pd.DataFrame([input_data])
    prediction = lr_model.predict(input_df)

@api.post('/predict')
async def predict(airline: Airline):
    review = airline.text
    prediction = lr_model.predict([review])
    return prediction
