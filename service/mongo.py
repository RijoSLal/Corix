from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from fastapi import FastAPI
from dotenv import load_dotenv
from pydantic import BaseModel,Field
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

# Pydantic model representing a new user for registration
class User(BaseModel):
     username : str 
     email : str 
     password : str 

# Model used for authenticating a user by username and password
class User_info(BaseModel):
     username : str 
     password : str


# Model for accepting clinical/lab parameters for prediction
class Lab_Params(BaseModel):
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

# Model for accepting wearable device signals for prediction
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


#Documentation of all endpoints and service
document = {
  "/register": {
    "description": "Register a new user",
    "input": {
      "username": "User's unique identifier",
      "email": "User's email address",
      "password": "User's password for authentication"
    }
  },
  "/user_info": {
    "description": "Get user information",
    "input": {
      "username": "User's unique identifier",
      "password": "User's password for authentication"
    }
  },
  "/predict/from-lab": {
    "description": "Predict health outcome using lab parameters",
    "input": {
      "ApiKey": "API key for authentication",
      "Age": "Patient's age in years",
      "Gender": "Patient's gender: 0 - female, 1 - male",
      "Height": "Patient's height in cm",
      "Weight": "Patient's weight in kg",
      "SystolicBP": "Systolic blood pressure",
      "DiastolicBP": "Diastolic blood pressure",
      "Cholesterol": "Cholesterol level: 1 - normal, 2 - above normal, 3 - well above normal",
      "Glucose": "Glucose level: 1 - normal, 2 - above normal, 3 - well above normal",
      "Smoke": "Smoking status: 0 - no, 1 - yes",
      "AlcoholIntake": "Alcohol intake: 0 - no, 1 - yes",
      "PhysicalActivity": "Physical activity: 0 - no, 1 - yes",
      "BMI": "Body Mass Index (calculated from height and weight)"
    }
  },
  "/predict/from-wearables": {
    "description": "Predict health outcome using wearable sensor data",
    "input": {
      "ApiKey": "API key for authentication",
      "PPG_1": "Photoplethysmogram channel 1",
      "PPG_2": "Photoplethysmogram channel 2",
      "PPG_3": "Photoplethysmogram channel 3",
      "ECG_1": "Electrocardiogram channel 1",
      "ECG_2": "Electrocardiogram channel 2",
      "ECG_3": "Electrocardiogram channel 3",
      "SCG_1": "Seismocardiogram channel 1",
      "SCG_2": "Seismocardiogram channel 2",
      "SCG_3": "Seismocardiogram channel 3",
      "BCG_1": "Ballistocardiogram channel 1",
      "BCG_2": "Ballistocardiogram channel 2",
      "BCG_3": "Ballistocardiogram channel 3"
    }
  }
}
