import pandas as pd
from src.data_preprocessing import Prepare_sentiment_data
from src.model_training import Training
import logging 
from src.model_pusher import model_pusher


logging.basicConfig(
     level = logging.DEBUG,
     format = "%(asctime)s - %(levelname)s - %(message)s"
)

def train_and_evaluate():
    try:
        train_dataset, test_dataset = Prepare_sentiment_data()
        train = Training()
        trainer = train.model_training(train_dataset= train_dataset, test_dataset= test_dataset)
        results = train.model_evaluation(trainer)
        
        print(results)
        Pusher = model_pusher()
        Pusher.updated_model_pusher(trainer, results)

    except Exception as e:
            logging.error(f"error occurred while training the model {e}")
       
train_and_evaluate()