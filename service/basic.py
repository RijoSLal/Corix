import asyncio
import secrets
import bcrypt
import numpy as np
import joblib
from datetime import datetime

LENGTH_API_KEY = 32

# load the scaler once on module import
scaler = joblib.load("model/scaler_used_for_lab_model.pkl")

async def create_api_key() -> str:
    """
    Generate a secure random API key using URL-safe base64 encoding.

    Returns:
        str: A secure API key string.
    """
    return await asyncio.to_thread(secrets.token_urlsafe, LENGTH_API_KEY)

async def encrypt_password(password: str) -> str:
    """
    Hash a plaintext password using bcrypt.

    Args:
        password (str): The plaintext password.

    Returns:
        str: The bcrypt hashed password.
    """
    def _encrypt():
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode(), salt)
        return hashed.decode()
    return await asyncio.to_thread(_encrypt)

async def verify_password(password: str, hashed: str) -> bool:
    """
    Verify if a plaintext password matches a bcrypt hashed password.

    Args:
        password (str): The plaintext password.
        hashed (str): The hashed password.

    Returns:
        bool: True if the password matches the hash, False otherwise.
    """
    return await asyncio.to_thread(
        lambda: bcrypt.checkpw(password.encode(), hashed.encode())
    )

async def is_eligible(api_key: str, collections) -> dict:
    """
    Check if a user exists with the provided API key.

    Args:
        api_key (str): The API key to verify.
        collections: The MongoDB collection to query.

    Returns:
        dict: The user document if found, otherwise None.
    """
    return await asyncio.to_thread(collections.find_one, {"api_key": api_key})

async def tokens_limit_and_userdata_logs(
    usage_log: list, usage_index: int, readings: list, prediction: list
) -> tuple:
    """
    Manage the usage log and rotate it on a circular buffer basis.

    Args:
        usage_log (list): List containing the last 12 logs.
        usage_index (int): The current index in the log.
        readings (list): Input data for prediction.
        prediction (list): Model's prediction result.

    Returns:
        tuple: (updated_log, next_index)
    """
    def _process():
        nonlocal usage_log, usage_index
        if usage_log is None or not isinstance(usage_log, list) or len(usage_log) != 12:
            usage_log = [{} for _ in range(12)]
            usage_index = 0

        usage_log[usage_index] = {
            "timestamp": datetime.utcnow().isoformat(),
            "prediction": prediction,
            "data": readings
        }

        next_index = (usage_index + 1) % 12
        return usage_log, next_index

    return await asyncio.to_thread(_process)

async def lab_data_preprocessing(readings: list) -> np.ndarray:
    """
    Preprocess lab data using a fitted scaler.

    Args:
        readings (list): Raw input features for lab model.

    Returns:
        np.ndarray: Scaled input data as a numpy array.
    """
    def _preprocess():
        x_scaled = scaler.transform(readings)
        x = np.array(x_scaled, dtype=np.float32)
        return x
    return await asyncio.to_thread(_preprocess)

async def wearable_data_preprocessing(data: list) -> list:
    """
    Dummy preprocessing for wearable data for future use

    Args:
        data (list): Input wearable data.

    Returns:
        list: The same data (placeholder).
    """
    return data
