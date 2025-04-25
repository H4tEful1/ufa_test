from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.model import predict
from config.logs_config import *


app = FastAPI()


class TextIn(BaseModel):
    text: str


@app.post("/predict")
def get_prediction(input: TextIn):
    logger_interface.info(f"Received request with text: {input.text[:100]}...")

    try:
        result = predict(input.text)

        if "error" in result:
            logger_interface.warning(f"Validation error: {result['error']}")
            raise HTTPException(status_code=400, detail=result["error"])

        logger_interface.info(f"Prediction successful: {result}")
        return result

    except Exception as e:
        logger_interface.error(f"Unexpected error during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
