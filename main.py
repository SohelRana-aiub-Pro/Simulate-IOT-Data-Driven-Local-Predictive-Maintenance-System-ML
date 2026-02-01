import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
import random
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import sessionmaker, declarative_base
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle



# ----------------------------
# Database setup
# ----------------------------
DB_FILE = "iot_maintenance.db"
DATABASE_URL = f"sqlite:///./{DB_FILE}"

# Check if database exists
db_exists = os.path.exists(DB_FILE)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class IoTMachineData(Base):
    __tablename__ = "iot_machine_data"
    id = Column(Integer, primary_key=True, index=True)
    machine_id = Column(String(50))
    temperature = Column(Float)
    vibration = Column(Float)
    pressure = Column(Float)
    status = Column(String(50))  # "OK" or "FAIL"

# If database does not exist, create tables
if not db_exists:
    Base.metadata.create_all(bind=engine)

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="Simulate IoT Data Driven - Local Predictive Maintenance System - ML")

# ----------------------------
# ML Model Training
# ----------------------------
MODEL_FILE = "iot_maintenance_model.pkl"

def train_model():
    db = SessionLocal()
    data = pd.read_sql("SELECT * FROM iot_machine_data", db.bind)
    db.close()

    if data.empty:
        return None

    X = data[["temperature", "vibration", "pressure"]]
    y = data["status"].apply(lambda x: 1 if x == "FAIL" else 0)

    model = LogisticRegression()
    model.fit(X, y)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    return model

def load_model():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            return pickle.load(f)
    return None

# ----------------------------
# API Endpoints
# ----------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h1>Simulate IoT Data Driven - Local Predictive Maintenance System - ML</h1>
    <p>Available endpoints:</p>
    <ul>
        <li>/simulate_data?machine_id=1&count=10</li>
        <li>/add_data?machine_id=1&temperature=75&vibration=0.02&pressure=30&status=OK</li>
        <li>/train_model</li>
        <li>/predict?temperature=80&vibration=0.05&pressure=28</li>
    </ul>
    """

# Fix favicon issue
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    # If you have a favicon.ico file in your project folder, serve it
    if os.path.exists("favicon.ico"):
        return FileResponse("favicon.ico")
    # Otherwise return empty response to avoid 404
    return HTMLResponse(content="", status_code=200)



@app.get("/add_data")
def add_data(machine_id: str, temperature: float, vibration: float, pressure: float, status: str):
    db = SessionLocal()
    entry = IoTMachineData(
        machine_id=machine_id,
        temperature=temperature,
        vibration=vibration,
        pressure=pressure,
        status=status
    )
    db.add(entry)
    db.commit()
    db.close()
    return {"message": "Data added successfully"}

@app.get("/simulate_data")
def simulate_data(machine_id: str, count: int = 10):
    """
    Simulate IoT sensor data for testing.
    """
    db = SessionLocal()
    for _ in range(count):
        temperature = random.uniform(60, 100)
        vibration = random.uniform(0.01, 0.1)
        pressure = random.uniform(25, 40)
        status = "FAIL" if (temperature > 90 or vibration > 0.08 or pressure < 28) else "OK"

        entry = IoTMachineData(
            machine_id=machine_id,
            temperature=temperature,
            vibration=vibration,
            pressure=pressure,
            status=status
        )
        db.add(entry)
    db.commit()
    db.close()
    return {"message": f"{count} simulated IoT records added for machine {machine_id}"}

@app.get("/train_model")
def train():
    model = train_model()
    if model:
        return {"message": "Model trained successfully"}
    else:
        return {"message": "No data available to train"}

@app.get("/predict")
def predict(temperature: float, vibration: float, pressure: float):
    model = load_model()
    if not model:
        return {"error": "Model not trained yet"}
    X_new = [[temperature, vibration, pressure]]
    prediction = model.predict(X_new)[0]
    status = "FAIL" if prediction == 1 else "OK"
    return {"prediction": status}