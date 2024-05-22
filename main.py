from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI
import joblib

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
    gender: str
    age: int

app = FastAPI()

# Load the trained model and scaler
model = joblib.load("svc_classifier_model.pkl")
scaler = joblib.load("scaler.pkl")

# Function to encode gender
def encode_gender(gender: str) -> int:
    if gender.lower() in ['male', 'm']:
        return 0
    elif gender.lower() in ['female', 'f']:
        return 1
    else:
        raise ValueError("Gender must be 'male' or 'female'")

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
        data.bmi,
        data.age,
        encode_gender(data.gender)
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
    # Manually check predictions
    test_samples = [
        [75, 115, 18.0, 72, 95, 85, 60, 97.5, 8, 12000, 2500, 22.0, 30, 0],  # Expected: good
        [90, 135, 35.0, 100, 115, 110, 75, 92.0, 5, 4000, 900, 30.0, 50, 1]  # Expected: bad
    ]

    # Normalize the input data
    test_samples_scaled = scaler.transform(test_samples)

    # Make predictions
    predictions = model.predict(test_samples_scaled)
    print(predictions)
    uvicorn.run(app, host="0.0.0.0", port=8000)