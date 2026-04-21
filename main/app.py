import logging
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import pandas as pd
import numpy as np
from pipeline.prediction import predict_sentiment
from pipeline.training import train_and_evaluate

logging.basicConfig(
     level = logging.DEBUG,
     format = "%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI(title="Shop ease sentiment API")

# pydantic schema for input dataset
class TextRequest(BaseModel):
    text: str

predictor = predict_sentiment()
logging.info(f"model successfully loaded")

app.host("/predict_sentiment"):
def predict_text(request: TextRequest):
    try:
        result = predictor.predict(request.text)
        print(result)
        top_label = max(result, key=lambda x: x["score"])
        return {"label": top_label["label"], "confidence": float(top_label["score"])}
    except Exception as e:
        logging.error(f"error occurred while predicting the sentiment {e}")
