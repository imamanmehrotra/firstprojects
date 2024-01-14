import os
import sys
import numpy as np
import pandas as pd
import pickle

from src.logger.logging import logging
from src.exception.exception import customexception

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok= True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise customexception(e,sys)


def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report={}
        for model_name,model in zip(models.keys(), models.values()):
            model.fit(X_train, y_train)
            y_test_pred =model.predict(X_test)
            test_r2 = r2_score(y_test,y_test_pred)
            test_mse = mean_squared_error(y_test, y_pred)
            test_rmse = test_mse**0.5
            report[model_name] = [test_r2, test_rmse]
    
        return report

    except Exception as e:
        logging.info('Exception occured during model training')
        raise customexception(e,sys)


def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info("Exception occured in loading the object")
        raise customexception(e,sys)
