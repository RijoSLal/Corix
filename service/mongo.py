from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from fastapi import FastAPI
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ConfigDict
from contextlib import asynccontextmanager
import os 
import certifi 
import logging

logger = logging.getLogger(__name__)

load_dotenv()

@asynccontextmanager
async def startup_client_db(app: FastAPI):
    """
    Initializes MongoDB client connection at application startup and
    safely closes the connection on shutdown.

    Args:
        app (FastAPI): The FastAPI application instance
    """
    uri = os.getenv("ATLAS_URI")
    app.client = MongoClient(uri, server_api=ServerApi('1'),  tls=True,
    tlsCAFile=certifi.where(),
    connectTimeoutMS=10000,
    socketTimeoutMS=10000,
    serverSelectionTimeoutMS=10000,
    tlsAllowInvalidCertificates=False)
    app.db = app.client["corix"]
    app.collections = app.db["validation_set"]
    logger.info("mongodb connection established")
    yield 
    app.client.close()
    logger.info("mongodb connection suspended")

# pydantic model representing a new user for registration
class User(BaseModel):
     username : str 
     email : str 
     password : str 

# model used for authenticating a user by username and password
class User_info(BaseModel):
     username : str 
     password : str


# model for accepting clinical/lab parameters for prediction
class Lab_Params(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    ApiKey: str = Field(..., description="API key for authentication")
    Age: int = Field(..., description="Patient's age in days")
    Gender: int = Field(..., description="Patient's gender: 0 - female, 1 - male")
    Height: int = Field(..., description="Patient's height in cm")
    Weight: float = Field(..., description="Patient's weight in kg")
    SystolicBP: int = Field(..., alias="ap_hi", description="Systolic blood pressure")
    DiastolicBP: int = Field(..., alias="ap_lo", description="Diastolic blood pressure")
    Cholesterol: int = Field(..., description="Cholesterol level: 1 - normal, 2 - above normal, 3 - well above normal")
    Glucose: int = Field(..., description="Glucose level: 1 - normal, 2 - above normal, 3 - well above normal")
    Smoke: int = Field(..., description="Smoking status: 0 - no, 1 - yes")
    AlcoholIntake: int = Field(..., alias="alco", description="Alcohol intake: 0 - no, 1 - yes")
    PhysicalActivity: int = Field(..., alias="active", description="Physical activity: 0 - no, 1 - yes")
    BMI: float = Field(..., description="Body Mass Index (calculated from height and weight)")

# model for accepting wearable device signals for prediction
class Wearable_Params(BaseModel):
    ApiKey: str = Field(..., description="API key for authentication")
    PPG_1: float = Field(..., description="Photoplethysmogram channel 1")
    PPG_2: float = Field(..., description="Photoplethysmogram channel 2")
    PPG_3: float = Field(..., description="Photoplethysmogram channel 3")
    ECG_1: float = Field(..., description="Electrocardiogram channel 1")
    ECG_2: float = Field(..., description="Electrocardiogram channel 2")
    ECG_3: float = Field(..., description="Electrocardiogram channel 3")
    SCG_1: float = Field(..., description="Seismocardiogram channel 1")
    SCG_2: float = Field(..., description="Seismocardiogram channel 2")
    SCG_3: float = Field(..., description="Seismocardiogram channel 3")
    BCG_1: float = Field(..., description="Ballistocardiogram channel 1")
    BCG_2: float = Field(..., description="Ballistocardiogram channel 2")
    BCG_3: float = Field(..., description="Ballistocardiogram channel 3") 


# documentation of all endpoints and service
document = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Corix API Documentation</title>
    <style>
        body {
            font-family: monospace;
            padding: 20px;
            max-width: 800px;
            margin: 0 auto;
            background-color: #ffffff;
            color: #000000;
            line-height: 1.4;
        }
        h1 {
            border-bottom: 1px solid #000;
            padding-bottom: 8px;
        }
        .endpoint {
            margin-bottom: 40px;
            border-bottom: 1px dashed #ccc;
            padding-bottom: 30px;
        }
        .endpoint:last-child {
            border-bottom: none;
        }
        .route {
            font-weight: bold;
            font-size: 1.2em;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin-top: 10px;
            margin-bottom: 20px;
        }
        th, td {
            border: 1px solid #000000;
            padding: 6px 10px;
            text-align: left;
        }
        th {
            background-color: #f0f0f0;
        }
        pre {
            background-color: #f5f5f5;
            border: 1px solid #ccc;
            padding: 10px;
            overflow-x: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
            margin-top: 5px;
            margin-bottom: 15px;
        }
        .example-title {
            font-weight: bold;
            margin-top: 15px;
            display: block;
        }
    </style>
</head>
<body>
    <h1>CORIX API DOCUMENTATION</h1>

    <div class="endpoint">
        <div class="route">POST /register</div>
        <p>Register a new user</p>
        <table>
            <thead>
                <tr>
                    <th style="width: 30%;">Parameter</th>
                    <th>Description</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>username</td>
                    <td>User's unique identifier</td>
                </tr>
                <tr>
                    <td>email</td>
                    <td>User's email address</td>
                </tr>
                <tr>
                    <td>password</td>
                    <td>User's password for authentication</td>
                </tr>
            </tbody>
        </table>

        <strong>Examples:</strong>

        <div class="example-title">cURL</div>
        <pre>curl -X POST http://localhost:8000/register \
  -H "Content-Type: application/json" \
  -d '{"username": "testuser", "email": "user@example.com", "password": "mypassword"}'</pre>

        <div class="example-title">Python (requests)</div>
        <pre>import requests

url = "http://localhost:8000/register"
payload = {
    "username": "testuser",
    "email": "user@example.com",
    "password": "mypassword"
}
response = requests.post(url, json=payload)
print(response.json())</pre>

        <div class="example-title">JavaScript (fetch)</div>
        <pre>fetch("http://localhost:8000/register", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    username: "testuser",
    email: "user@example.com",
    password: "mypassword"
  })
})
.then(res => res.json())
.then(console.log);</pre>
    </div>

    <div class="endpoint">
        <div class="route">POST /user_info</div>
        <p>Get user information</p>
        <table>
            <thead>
                <tr>
                    <th style="width: 30%;">Parameter</th>
                    <th>Description</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>username</td>
                    <td>User's unique identifier</td>
                </tr>
                <tr>
                    <td>password</td>
                    <td>User's password for authentication</td>
                </tr>
            </tbody>
        </table>

        <strong>Examples:</strong>

        <div class="example-title">cURL</div>
        <pre>curl -X POST http://localhost:8000/user_info \
  -H "Content-Type: application/json" \
  -d '{"username": "testuser", "password": "mypassword"}'</pre>

        <div class="example-title">Python (requests)</div>
        <pre>import requests

url = "http://localhost:8000/user_info"
payload = {
    "username": "testuser",
    "password": "mypassword"
}
response = requests.post(url, json=payload)
print(response.json())</pre>

        <div class="example-title">JavaScript (fetch)</div>
        <pre>fetch("http://localhost:8000/user_info", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    username: "testuser",
    password: "mypassword"
  })
})
.then(res => res.json())
.then(console.log);</pre>
    </div>

    <div class="endpoint">
        <div class="route">POST /predict/from-lab</div>
        <p>Predict health outcome using lab parameters</p>
        <table>
            <thead>
                <tr>
                    <th style="width: 30%;">Parameter</th>
                    <th>Description</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>ApiKey</td>
                    <td>API key for authentication</td>
                </tr>
                <tr>
                    <td>Age</td>
                    <td>Patient's age in days</td>
                </tr>
                <tr>
                    <td>Gender</td>
                    <td>Patient's gender: 0 - female, 1 - male</td>
                </tr>
                <tr>
                    <td>Height</td>
                    <td>Patient's height in cm</td>
                </tr>
                <tr>
                    <td>Weight</td>
                    <td>Patient's weight in kg</td>
                </tr>
                <tr>
                    <td>SystolicBP</td>
                    <td>Systolic blood pressure (can also be passed as <code>ap_hi</code>)</td>
                </tr>
                <tr>
                    <td>DiastolicBP</td>
                    <td>Diastolic blood pressure (can also be passed as <code>ap_lo</code>)</td>
                </tr>
                <tr>
                    <td>Cholesterol</td>
                    <td>Cholesterol level: 1 - normal, 2 - above normal, 3 - well above normal</td>
                </tr>
                <tr>
                    <td>Glucose</td>
                    <td>Glucose level: 1 - normal, 2 - above normal, 3 - well above normal</td>
                </tr>
                <tr>
                    <td>Smoke</td>
                    <td>Smoking status: 0 - no, 1 - yes</td>
                </tr>
                <tr>
                    <td>AlcoholIntake</td>
                    <td>Alcohol intake: 0 - no, 1 - yes (can also be passed as <code>alco</code>)</td>
                </tr>
                <tr>
                    <td>PhysicalActivity</td>
                    <td>Physical activity: 0 - no, 1 - yes (can also be passed as <code>active</code>)</td>
                </tr>
                <tr>
                    <td>BMI</td>
                    <td>Body Mass Index (calculated from height and weight)</td>
                </tr>
            </tbody>
        </table>

        <strong>Examples:</strong>

        <div class="example-title">cURL</div>
        <pre>curl -X POST http://localhost:8000/predict/from-lab \
  -H "Content-Type: application/json" \
  -d '{"ApiKey": "your_api_key", "Age": 18000, "Gender": 1, "Height": 175, "Weight": 70.0, "ap_hi": 120, "ap_lo": 80, "Cholesterol": 1, "Glucose": 1, "Smoke": 0, "alco": 0, "active": 1, "BMI": 22.9}'</pre>

        <div class="example-title">Python (requests)</div>
        <pre>import requests

url = "http://localhost:8000/predict/from-lab"
payload = {
    "ApiKey": "your_api_key",
    "Age": 18000,
    "Gender": 1,
    "Height": 175,
    "Weight": 70.0,
    "ap_hi": 120,
    "ap_lo": 80,
    "Cholesterol": 1,
    "Glucose": 1,
    "Smoke": 0,
    "alco": 0,
    "active": 1,
    "BMI": 22.9
}
response = requests.post(url, json=payload)
print(response.json())</pre>

        <div class="example-title">JavaScript (fetch)</div>
        <pre>fetch("http://localhost:8000/predict/from-lab", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    ApiKey: "your_api_key",
    Age: 18000,
    Gender: 1,
    Height: 175,
    Weight: 70.0,
    ap_hi: 120,
    ap_lo: 80,
    Cholesterol: 1,
    Glucose: 1,
    Smoke: 0,
    alco: 0,
    active: 1,
    BMI: 22.9
  })
})
.then(res => res.json())
.then(console.log);</pre>
    </div>

    <div class="endpoint">
        <div class="route">POST /predict/from-wearables</div>
        <p>Predict health outcome using wearable sensor data</p>
        <table>
            <thead>
                <tr>
                    <th style="width: 30%;">Parameter</th>
                    <th>Description</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>ApiKey</td>
                    <td>API key for authentication</td>
                </tr>
                <tr>
                    <td>PPG_1</td>
                    <td>Photoplethysmogram channel 1</td>
                </tr>
                <tr>
                    <td>PPG_2</td>
                    <td>Photoplethysmogram channel 2</td>
                </tr>
                <tr>
                    <td>PPG_3</td>
                    <td>Photoplethysmogram channel 3</td>
                </tr>
                <tr>
                    <td>ECG_1</td>
                    <td>Electrocardiogram channel 1</td>
                </tr>
                <tr>
                    <td>ECG_2</td>
                    <td>Electrocardiogram channel 2</td>
                </tr>
                <tr>
                    <td>ECG_3</td>
                    <td>Electrocardiogram channel 3</td>
                </tr>
                <tr>
                    <td>SCG_1</td>
                    <td>Seismocardiogram channel 1</td>
                </tr>
                <tr>
                    <td>SCG_2</td>
                    <td>Seismocardiogram channel 2</td>
                </tr>
                <tr>
                    <td>SCG_3</td>
                    <td>Seismocardiogram channel 3</td>
                </tr>
                <tr>
                    <td>BCG_1</td>
                    <td>Ballistocardiogram channel 1</td>
                </tr>
                <tr>
                    <td>BCG_2</td>
                    <td>Ballistocardiogram channel 2</td>
                </tr>
                <tr>
                    <td>BCG_3</td>
                    <td>Ballistocardiogram channel 3</td>
                </tr>
            </tbody>
        </table>

        <strong>Examples:</strong>

        <div class="example-title">cURL</div>
        <pre>curl -X POST http://localhost:8000/predict/from-wearables \
  -H "Content-Type: application/json" \
  -d '{"ApiKey": "your_api_key", "PPG_1": 0.5, "PPG_2": 0.5, "PPG_3": 0.5, "ECG_1": 0.5, "ECG_2": 0.5, "ECG_3": 0.5, "SCG_1": 0.5, "SCG_2": 0.5, "SCG_3": 0.5, "BCG_1": 0.5, "BCG_2": 0.5, "BCG_3": 0.5}'</pre>

        <div class="example-title">Python (requests)</div>
        <pre>import requests

url = "http://localhost:8000/predict/from-wearables"
payload = {
    "ApiKey": "your_api_key",
    "PPG_1": 0.5, "PPG_2": 0.5, "PPG_3": 0.5,
    "ECG_1": 0.5, "ECG_2": 0.5, "ECG_3": 0.5,
    "SCG_1": 0.5, "SCG_2": 0.5, "SCG_3": 0.5,
    "BCG_1": 0.5, "BCG_2": 0.5, "BCG_3": 0.5
}
response = requests.post(url, json=payload)
print(response.json())</pre>

        <div class="example-title">JavaScript (fetch)</div>
        <pre>fetch("http://localhost:8000/predict/from-wearables", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    ApiKey: "your_api_key",
    PPG_1: 0.5, PPG_2: 0.5, PPG_3: 0.5,
    ECG_1: 0.5, ECG_2: 0.5, ECG_3: 0.5,
    SCG_1: 0.5, SCG_2: 0.5, SCG_3: 0.5,
    BCG_1: 0.5, BCG_2: 0.5, BCG_3: 0.5
  })
})
.then(res => res.json())
.then(console.log);</pre>
    </div>
</body>
</html>"""
