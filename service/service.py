from fastapi import FastAPI ,HTTPException , status
from fastapi.responses import JSONResponse, HTMLResponse
from mongo import startup_client_db,User,Lab_Params,Wearable_Params,User_info,document
import logging 
from datetime import datetime
from model_retriever import Model_Fetch 
import os
from dotenv import load_dotenv
from basic import (
    create_api_key,
    encrypt_password,
    verify_password,
    is_eligible,
    tokens_limit_and_userdata_logs,
    lab_data_preprocessing,
    wearable_data_preprocessing
)
from redis_client import redis_client
import json
import asyncio

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

app = FastAPI(lifespan=startup_client_db)

logger = logging.getLogger(__name__)


load_dotenv()

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
lab_model_name = os.getenv("LAB_MODEL_NAME")
lab_model_version = os.getenv("LAB_MODEL_VERSION")
wearable_model_name = os.getenv("WEARABLE_MODEL_NAME")
wearable_model_version = os.getenv("WEARABLE_MODEL_VERSION")

FETCH_MODELS = Model_Fetch(tracking_uri)

LAB_MODEL = FETCH_MODELS.load_lab_model(lab_model_name,lab_model_version)
WEAR_MODEL = FETCH_MODELS.load_wearable_model(wearable_model_name,wearable_model_version)

@app.get("/", response_class=HTMLResponse)
async def documentation() -> HTMLResponse:
    """
    Endpoint that provides HTML documentation for the Corix Service API.

    Returns:
        HTMLResponse: Rendered HTML API documentation.
    """
    return HTMLResponse(content=document, status_code=200)

@app.post("/register")
async def register_device(user : User) -> JSONResponse: # realistically checking username is not a common practice but i am too lazy to remove it
        """
        Register a new user and generate an API key.

        Args:
            user (User): User registration data.

        Returns:
            JSONResponse: Contains the API key if registration is successful.
        """
        # check username existence in redis first
        username_exists = await redis_client.get(f"username:exists:{user.username}")
        if username_exists:
            raise HTTPException(status_code=400, detail="username already exist")
            
        # check email existence in redis first
        email_exists = await redis_client.get(f"email:exists:{user.email}")
        if email_exists:
            raise HTTPException(status_code=400, detail="email already exist")

        # fallback to mongodb check if not found in cache
        db_user = await asyncio.to_thread(app.collections.find_one, {"username": user.username})
        if db_user:
            await redis_client.setex(f"username:exists:{user.username}", 3600, "1")
            raise HTTPException(status_code=400, detail="username already exist")
            
        db_email = await asyncio.to_thread(app.collections.find_one, {"email": user.email})
        if db_email:
            await redis_client.setex(f"email:exists:{user.email}", 3600, "1")
            raise HTTPException(status_code=400, detail="email already exist")
        
        api_key = await create_api_key()
        hased_password = await encrypt_password(user.password)
        user_data = {
             "username": user.username,
             "email":user.email,
             "password":hased_password,
             "api_key":api_key
        }

        # insert user to database
        insert_result = await asyncio.to_thread(app.collections.insert_one, user_data)
        
        # cache profile info in redis (with ttl 1hr = 3600s)
        user_to_cache = user_data.copy()
        if "_id" in user_to_cache:
            user_to_cache["_id"] = str(user_to_cache["_id"])
        await redis_client.setex(f"user:profile:{user.username}", 3600, json.dumps(user_to_cache))
        
        # cache existence checks in redis
        await redis_client.setex(f"username:exists:{user.username}", 3600, "1")
        await redis_client.setex(f"email:exists:{user.email}", 3600, "1")
        
        return JSONResponse(content={"api_key" : api_key},status_code=status.HTTP_201_CREATED)

@app.post("/user_info")
async def user_details_database_fetch(user_info: User_info) -> JSONResponse:
    """
    Authenticate user by checking username and hashed password.

    Args:
        user_info (User_info): User credentials.

    Returns:
        JSONResponse: User data if authenticated, otherwise error.
    """
    # check redis cache first
    cache_key = f"user:profile:{user_info.username}"
    cached_user = await redis_client.get(cache_key)
    
    if cached_user:
        user = json.loads(cached_user)
    else:
        # fallback to mongodb
        user = await asyncio.to_thread(app.collections.find_one, {"username": user_info.username})
        if user:
            # stringify objectid to make it json serializable for caching
            user_to_cache = user.copy()
            if "_id" in user_to_cache:
                user_to_cache["_id"] = str(user_to_cache["_id"])
            await redis_client.setex(cache_key, 3600, json.dumps(user_to_cache))
            
    if not user:
        return JSONResponse(
            status_code=401,
            content={
                "success": False,
                "error": "Unauthorized access"
            }
        )

    verified = await verify_password(user_info.password, user["password"])
    
    # prepare sanitized dict to return
    user_data = user.copy()
    user_data.pop("_id", None)  
    user_data.pop("password", None)
    
    if verified:
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "user_info": user_data
            }
        )
    else:
        return JSONResponse(
            status_code=401,
            content={
                "success": False,
                "error": "Unauthorized access"
            }
        )

@app.post("/predict/from-wearables")
async def prediction_from_wearables(Wearable_P : Wearable_Params) -> JSONResponse:
    """
    Generate a prediction based on wearable sensor data.

    Args:
        Wearable_P (Wearable_Params): Wearable input parameters including API key.

    Returns:
        JSONResponse: Model prediction or unauthorized error.
    """
    
    user = await is_eligible(Wearable_P.ApiKey, app.collections)
    if user:
        usage_index = user.get("usage_index", 0)

        fields = [
            "PPG_1", "PPG_2", "PPG_3",
            "ECG_1", "ECG_2", "ECG_3",
            "SCG_1", "SCG_2", "SCG_3",
            "BCG_1", "BCG_2", "BCG_3"
        ]
        
       
        readings = [[getattr(Wearable_P, field) for field in fields]]
        prediction = WEAR_MODEL.predict(readings)

        usage_log = user.get("user_wearable_log")
        usage_index = user.get("usage_index", 0)

        updated_log, next_index = await tokens_limit_and_userdata_logs(
            usage_log, usage_index, readings, prediction.tolist()
        )

        await asyncio.to_thread(
            app.collections.update_one,
            {"api_key": Wearable_P.ApiKey},
            {
                "$set": {
                    "user_wearable_log": updated_log,
                    "usage_index": next_index
                }
            }
        )
        
        HEART_DISEASE_LABELS = ("No potential heart disease", "Potential heart disease")
        PREDICTION = HEART_DISEASE_LABELS[
                     prediction.tolist()[0]
                     ]
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "prediction": PREDICTION
            }
        )  
    else:
        return JSONResponse(
        status_code=401,
        content={
            "success": False,
            "error": "Unauthorized access"
        }
    )

@app.post("/predict/from-lab")
async def prediction_from_lab(Lab_P : Lab_Params):
    """
    Generate a prediction based on lab data.

    Args:
        Lab_P (Lab_Params): Lab input parameters including API key.

    Returns:
        JSONResponse: Model prediction or unauthorized error.
    """
    user = await is_eligible(Lab_P.ApiKey, app.collections)
    if user:

        fields = [
            "Age",
            "Gender",
            "Height",
            "Weight",
            "SystolicBP",
            "DiastolicBP",
            "Cholesterol",
            "Glucose",
            "Smoke",
            "AlcoholIntake",
            "PhysicalActivity", 
            "BMI"
        ]
        

        readings = [[getattr(Lab_P, field) for field in fields]]
        preprocessed = await lab_data_preprocessing(readings)
        probability = LAB_MODEL.predict(preprocessed)
        threshold = 0.5
        prediction = int(probability[0][0] >= threshold)

        if prediction == 1:
            explanation = (
                f"The model predicts a {probability[0][0]:.1%} likelihood of heart disease. "
                "This suggests a potential health risk. Consider consulting a medical professional for further evaluation."
            )
        else:
            explanation = (
                f"The model predicts only a {probability[0][0]:.1%} likelihood of heart disease. "
                "This indicates no current signs of potential heart disease based on the provided data."
            )


        user_lab = {
              "timestamp": datetime.utcnow().isoformat(),
              "prediction": prediction,
        }

        await asyncio.to_thread(
            app.collections.update_one,
            {"api_key": Lab_P.ApiKey},
            {
                "$set": {
                    "user_lab": user_lab,
                }
            }
        )


        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "prediction": explanation
            }
        )
    else:
        return JSONResponse(
        status_code=401,
        content={
            "success": False,
            "error": "Unauthorized access"
        }
    )
