from pathlib import Path
import pandas as pd
import numpy as np
import re
import logging
from config.constant import Input_Data

logging.basicConfig(level=logging.INFO)

def data_injection():
    try:
        data = pd.read_csv(Input_Data)
        logging.info("Data successfully loaded")
        print(data.head(5))
        return data
    except Exception as e:
        logging.error(f"error occurred while loading the data{e}")
data_injection()