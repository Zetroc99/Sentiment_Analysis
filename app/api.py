import pickle
import SentimentModel

from fastapi import FastAPI
from pydantic import BaseModel

# Initialize API
api = FastAPI()

# Load logistic regression model
pkl_filename = '../model/airline_model.pkl'
with open(pkl_filename, 'rb') as file:
    lr_model = pickle.load(file)


# Data validation
class Airline(BaseModel):
    text: str


# Test endpoint
@api.get('/')
def root():
    return {'message': 'Hello!'}


@api.post('/predict')
async def predict(airline: Airline):
    review = airline.dict()['text']
    prediction = lr_model.predict([review])
    return str(prediction[0])