import pandas as pd
import numpy as np
from src.logger.logging import logging
from src.exception.exception import customexception
import os
import sys
from dataclass import dataclass
from pathlib import path
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso
from src.utils.utils import load_object, save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    pass

class ModelTrainer:
    def __init__(self):
        pass

    def initiate_model_training(self):
        try:
            pass
        except Exception as e:
            logging.info()
            raise customexception(e,sys)
