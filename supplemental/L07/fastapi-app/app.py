from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hello from Docker!", "status": "running"}

@app.get("/predict")
def predict():
    # In a real app, this would run your ML model
    return {"prediction": 0.87, "confidence": "high"}