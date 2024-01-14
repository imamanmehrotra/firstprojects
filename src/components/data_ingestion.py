import pandas as pd
import numpy as np
from src.logger.logging import logging
from src.exception.exception import customexception

import os
import sys
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path
sys.path.insert(0, '../src')


@dataclass
class DataIngestionConfig:
    raw_data_path:str = os.path.join("artifacts", "raw.csv") 
    train_data_path:str = os.path.join("artifacts","train.csv")
    test_data_path:str = os.path.join("artifacts", "test.csv")
    

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Started")
        try:
            path = "https://raw.githubusercontent.com/sunnysavita10/fsdsmendtoend/main/notebooks/data/gemstone.csv"
            data = pd.read_csv(path)
            logging.info("Reading the dataframe")

            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)),exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info("Saved the raw dataset in artifacts folder")

            train_data,test_data = train_test_split(data,test_size = 0.25,random_state = 42)
            logging.info("Train Test split completed")

            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, index = False)
            
            logging.info("Data Ingestion Completed")

            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)
        
        except Exception as e:
            logging.info()
            raise customexception(e,sys)



    # def initiate_data_ingestion(self):
    #     path = "https://raw.githubusercontent.com/sunnysavita10/fsdsmendtoend/main/notebooks/data/gemstone.csv"
    #     data = pd.read_csv(path)

    #     os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)),exist_ok=True)
    #     data.to_csv(self.ingestion_config.raw_data_path,index=False)

    #     train_data,test_data = train_test_split(data,test_size = 0.25,random_state = 42)

    #     train_data.to_csv(self.ingestion_config.train_data_path, index=False)
    #     test_data.to_csv(self.ingestion_config.test_data_path, index = False)
        
    #     return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)

        

if __name__=="__main__":
    obj = DataIngestion()

    obj.initiate_data_ingestion()
