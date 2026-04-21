import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer 
import transformers 
from transformers import Trainer, TrainingArguments
import os
from config.constant import Train_Data, Test_Data, model_name, training_args, num_of_labels
from src.data_cleaning import clean_data
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import logging
from src.data_preprocessing import Prepare_sentiment_data

#configure logging
logging.basicConfig(level=logging.INFO)

class Training:
    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = num_of_labels)
    
    def compute_metrics(self, p):
        preds = np.argmax(p.predictions, axis=1)
        labels = p.label_ids
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="weighted")
        return {"accuracy": acc, "f1": f1}
    
    def model_training(self, train_dataset, test_dataset):
        try:
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                compute_metrics=self.compute_metrics,
            )
            trainer.train()
            logging.info(f"model has been successfully trained..")
            return trainer
        except Exception as e:
            logging.error(f"error occurred while training the model {e}")

    def model_evaluation(self, trainer):
        results = trainer.evaluate()
        return results
    

