from transformers import AutoTokenizer
from transformers import BertTokenizer
import pandas as pd
import numpy as np
import torch
import logging
import os
from sklearn.model_selection import train_test_split
from config.constant import Cleaned_Data, model_name, truncation, padding, max_length, Train_Data, Input_Data, Test_Data
from src.data_cleaning import clean_data

class data_processor:
    def __init__(self):
        self.data = clean_data(pd.read_csv(Input_Data))
    
    def split_data(self):
        try:   
            x = self.data["final_text"].astype(str)
            y = self.data["label"].astype(int)
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.2, random_state=42
            )
            logging.info(f"Data successfully splitted")
            return x_train, x_test, y_train, y_test
        except Exception as e:
            logging.error(f"error occurred while spliting the data {e}")

class Tokenizer:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def encode(self, text):
        return self.tokenizer(
            text.to_list(),
            truncation = truncation,
            padding = padding,
            max_length = max_length

        )
    
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        #Ensure labels are a list without pandas index issues
        if hasattr(labels, "tolist"):
            self.labels = labels.tolist()
        elif hasattr(labels, "__iter__") and not isinstance(list, tuple):
            self.labels = list(labels)
        else:
            self.labels = labels

    def __len__(self):
      return len(self.labels)

    def __getitem__(self, idx):
      item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
      item["labels"] = torch.tensor(self.labels[idx])
      return item


def Prepare_sentiment_data():
    try:
        processor = data_processor()
        x_train, x_test, y_train, y_test = processor.split_data()
        if hasattr(y_train, "tolist"):
            y_train = y_train.tolist()
        if hasattr(y_test, "tolist"):
            y_test = y_test.tolist()
        tokenizer = Tokenizer()
        train_encodings = tokenizer.encode(x_train)
        test_encodings = tokenizer.encode(x_test)
        train_dataset = SentimentDataset(train_encodings, y_train)
        test_dataset = SentimentDataset(test_encodings, y_test)
        logging.info(f"Dataset has been successfully prepared..")
        os.makedirs(os.path.dirname(Train_Data), exist_ok=True)
        os.makedirs(os.path.dirname(Test_Data), exist_ok=True)
        torch.save(train_dataset, Train_Data)
        torch.save(test_dataset, Test_Data)
        return train_dataset, test_dataset
        
    except Exception as e:
        logging.error(f"error occurred while preparing the dataset {e}")

#Prepare_sentiment_data()
                  

        
