from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI
import joblib
import numpy as np

class UserData(BaseModel):
    diastolic: int
    systolic: int
    body_fat_rate: float
    heartrate_avg: float
    heartrate_count: int
    heartrate_max: int
    heartrate_min: int
    oxygen_saturation_value: float
    sleep_time: float
    steps_count: int
    water_volume: float
    bmi: float

app = FastAPI()

# Load the trained model and scaler
model = joblib.load("logistic_regressor_model.pkl")
scaler = joblib.load("scaler.pkl")  # Assuming you saved the scaler with this filename

@app.post('/predict')
def predict(data: UserData):
    input_data = [
        data.diastolic,
        data.systolic,
        data.body_fat_rate,
        data.heartrate_avg,
        data.heartrate_count,
        data.heartrate_max,
        data.heartrate_min,
        data.oxygen_saturation_value,
        data.sleep_time,
        data.steps_count,
        data.water_volume,
        data.bmi
    ]

    # Normalize the input data
    input_data_scaled = scaler.transform([input_data])

    # Make a prediction
    prediction = model.predict(input_data_scaled)
    prediction = prediction[0]  # Get the first prediction
    prediction = int(prediction)  # Convert numpy.int64 to int
    
    return {
        'prediction': prediction
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)