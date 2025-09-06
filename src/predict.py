from fastapi import FastAPI
import joblib
import numpy as np

# Import the data validation schema we just created
from src.schemas import IrisFeatures

# 1. Initialize the FastAPI app
# THIS LINE IS THE MOST IMPORTANT. The variable must be named 'app'
app = FastAPI(title="Iris Classifier API", version="1.0")

# 2. Load the trained model
# Ensure this path is correct. It points to the model from the project root.
model = joblib.load("models/iris_rf_model.joblib")

# 3. Define a simple root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Classifier API"}

# 4. Define the main prediction endpoint
@app.post("/predict")
def predict_species(flower: IrisFeatures):
    """
    Predicts the iris species from input measurements.
    """
    # Convert the incoming data into a numpy array for the model
    data = np.array([
        [flower.sepal_length, flower.sepal_width, 
         flower.petal_length, flower.petal_width]
    ])
    
    # Get the model's prediction
    prediction = model.predict(data)
    
    # Map the numerical prediction (0, 1, 2) to the actual species name
    species = ['setosa', 'versicolor', 'virginica']
    predicted_species = species[prediction[0]]
    
    # Return the prediction as a JSON response
    return {"predicted_species": predicted_species}